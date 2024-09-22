import argparse
import torch
from transformers import (
    AutoModelForSequenceClassification,
    Phi3ForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, TaskType, get_peft_model
from definition_handler.process_data import DatasetsHandler
import pandas as pd
import re
from datasets import Dataset, load_dataset
import evaluate
import numpy as np
from huggingface_hub import login
from accelerate import Accelerator


def compute_metrics(eval_pred):
    # All metrics are already predefined in the HF `evaluate` package
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric= evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")

    logits, labels = eval_pred # eval_pred is the tuple of predictions and labels returned by the model
    predictions = np.argmax(logits, axis=-1)
    precision = precision_metric.compute(predictions=predictions, references=labels)["precision"]
    recall = recall_metric.compute(predictions=predictions, references=labels)["recall"]
    f1 = f1_metric.compute(predictions=predictions, references=labels)["f1"]
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    # The trainer is expecting a dictionary where the keys are the metrics names and the values are the scores.
    return {"precision": precision, "recall": recall, "f1-score": f1, 'accuracy': accuracy}


################################################################################
# Custom loss function
################################################################################

class WeightedCELossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # Get model's predictions
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # Compute custom loss
        # loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor([neg_weights, pos_weights], device=model.device, dtype=logits.dtype))
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


################################################################################
# LoRA parameters
################################################################################
# LoRA attention dimension
# lora_r = 64
lora_r = 8
# Alpha parameter for LoRA scaling
lora_alpha = 16
# Dropout probability for LoRA layers
lora_dropout = 0.1

################################################################################
# bitsandbytes parameters
################################################################################
# Activate 4-bit precision base model loading
use_4bit = True
# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"
# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"
# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

################################################################################
# TrainingArguments parameters
################################################################################
# Number of training epochs
num_train_epochs = 1
# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = True
# Batch size per GPU for training
per_device_train_batch_size = 2
# Batch size per GPU for evaluation
per_device_eval_batch_size = 2
# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 4
# Enable gradient checkpointing
gradient_checkpointing = True
# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3
# Initial learning rate (AdamW optimizer)
# learning_rate = 2e-4
learning_rate = 2e-5
# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001
# Learning rate schedule (constant a bit better than cosine)
lr_scheduler_type = "constant"
# Number of training steps (overrides num_train_epochs)
max_steps = -1
# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03
# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True
# Save checkpoint every X updates steps
save_steps = 2500
# Log every X updates steps
logging_steps = 10

################################################################################
# SFT parameters
################################################################################
# Maximum sequence length to use
max_seq_length = 1664
# Pack multiple short examples in the same input sequence to increase efficiency
packing = True  # False
# Load the entire model on the GPU 0
# device_map = {"": 0}
# device_map = "auto"
# device_map = "cuda"
device_index = Accelerator().process_index
device_map = {"": device_index}

################################################################################
# methods
################################################################################

orca_template_with_def = """<|im_start|>system
You are MistralScico, a large language model trained by Tom Hope AI Lab. 
You will get two scientific texts that has a term surrounded by a relevant context and a definition of those terms that was generated with the context in mind. Read the terms with their context and definitions and define the correct relationship between the two terms as follows:
1 - Co-referring terms: Both term1 and term2 refer to the same underlying concept or entity.
2 - Parent concept: Term1 represents a broader category or concept that encompasses term2, such that mentioning term1 implicitly invokes term2.
3 - Child concept: The inverse of a parent concept relation. Term1 is a specific instance or subset of the broader concept represented by term2, such that mentioning term2 implicitly invokes term1.
0 - None of the above: Term1 and term2 are not co-referring, and do not have a parent-child or child-parent relation.
when answering the following question, please consider the context of the terms and their definitions and write out your reasoning step-by-step to be sure you get the right answers!
<|im_end|>
<|im_start|>user
here are the terms and their context:
first term: {term1}
first term definition: {term1_def}
first term context: {term1_text}
second term: {term2}
second term definition: {term2_def}
second term context: {term2_text}
please select the correct relationship between the two terms from the options above.<|im_end|>
<|im_start|>assistant
"""

phi3_instruct_prompt = """<|user|>
You are a helpful AI assistant. you will get two scientific texts that has a term surrounded by a relevant context. Read the terms with their context and define the correct relationship between the two terms as follows:
1 - Co-referring terms: Both term1 and term2 refer to the same underlying concept or entity.
2 - Parent concept: Term1 represents a broader category or concept that encompasses term2, such that mentioning term1 implicitly invokes term2.
3 - Child concept: The inverse of a parent concept relation. Term1 is a specific instance or subset of the broader concept represented by term2, such that mentioning term2 implicitly invokes term1.
0 - None of the above: Term1 and term2 are not co-referring, and do not have a parent-child or child-parent relation.
here are the terms and their context:
first term: {term1} 
first term context: {term1_text}
second term: {term2}
second term context: {term2_text}
please select the correct relationship between the two terms from the options above.<|end|>
<|assistant|>
"""

phi3_instruct_prompt_with_def = """<|user|>
You are a helpful AI assistant. you will get two scientific texts that has a term surrounded by a relevant context and a definition of those terms that was generated in regard for the context. Read the terms with their context and definitions and define the correct relationship between the two terms as follows:
1 - Co-referring terms: Both term1 and term2 refer to the same underlying concept or entity.
2 - Parent concept: Term1 represents a broader category or concept that encompasses term2, such that mentioning term1 implicitly invokes term2.
3 - Child concept: The inverse of a parent concept relation. Term1 is a specific instance or subset of the broader concept represented by term2, such that mentioning term2 implicitly invokes term1.
0 - None of the above: Term1 and term2 are not co-referring, and do not have a parent-child or child-parent relation.
here are the terms and their context:
first term: {term1}
first term definition: {term1_def}
first term context: {term1_text}
second term: {term2}
second term definition: {term2_def}
second term context: {term2_text}
please select the correct relationship between the two terms from the options above.<|end|>
<|assistant|>
"""

def get_phi3_instruct_prompt(pair, with_def = False, def_dict = None):
    term1_text, term2_text, _ = pair.split('</s>')
    term1 = re.search(r'<m>(.*?)</m>', term1_text).group(1).strip()
    term2 = re.search(r'<m>(.*?)</m>', term2_text).group(1).strip()
    term1_text, term2_text = term1_text.replace('<m> ', '').replace(' </m>', ''), term2_text.replace('<m> ',
                                                                                                     '').replace(
        ' </m>', '')

    if with_def:
        term1_def, term2_def = def_dict[pair.split('</s>')[0] + '</s>'], def_dict[pair.split('</s>')[1] + '</s>']
        return phi3_instruct_prompt_with_def.format(term1=term1, term1_text=term1_text, term2=term2,
                                                    term2_text=term2_text, term1_def=term1_def, term2_def=term2_def)

    return phi3_instruct_prompt.format(term1=term1, term1_text=term1_text, term2=term2, term2_text=term2_text)

def get_orca_format_prompt(pair, def_dict = None):
    term1_text, term2_text, _ = pair.split('</s>')
    term1 = re.search(r'<m>(.*?)</m>', term1_text).group(1).strip()
    term2 = re.search(r'<m>(.*?)</m>', term2_text).group(1).strip()
    term1_text, term2_text = term1_text.replace('<m> ', '').replace(' </m>', ''), term2_text.replace('<m> ',
                                                                                                     '').replace(
        ' </m>', '')

    term1_def, term2_def = def_dict[pair.split('</s>')[0] + '</s>'], def_dict[pair.split('</s>')[1] + '</s>']
    return orca_template_with_def.format(term1=term1, term1_text=term1_text, term2=term2,
                                                term2_text=term2_text, term1_def=term1_def, term2_def=term2_def)

def get_classification_lora_config(target_modules=["o_proj", "qkv_proj", "gate_up_proj", "down_proj"]):
    # Load LoRA configuration
    return LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        target_modules=target_modules,
        inference_mode=False,
        modules_to_save=["score"],
    )


def get_phi3_model_and_tokenizer(base_model, bnb_config):
    model = Phi3ForSequenceClassification.from_pretrained(
        base_model,
        # quantization_config=bnb_config,
        device_map=device_map,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        num_labels=4,
        torch_dtype=torch.bfloat16,
        # use_cache = False
    )

    model = get_peft_model(model, get_classification_lora_config())
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, add_prefix_space=True)
    tokenizer.pad_token = tokenizer.unk_token  # use unk rather than eos token to prevent endless generation
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = 'left'

    return model, tokenizer


def get_model_and_tokenizer(base_model, bnb_config):
    print(f'Loading model and tokenizer from {base_model}')
    if "Phi-3" in base_model:
        return get_phi3_model_and_tokenizer(base_model, bnb_config)
    else:
        # for now orca model
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map=device_map,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
            num_labels=4,
            torch_dtype=torch.bfloat16,
            # use_cache = False
        )

        lora_config = get_classification_lora_config(["q_proj", "up_proj", "o_proj", "k_proj", "down_proj", "gate_proj", "v_proj"])

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, add_prefix_space=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = 'left'
        model.config.pad_token_id = model.config.eos_token_id
        return model, tokenizer

def get_prompt_formatter(base_model):
    if "Phi-3" in base_model:
        return get_phi3_instruct_prompt
    else:
        return get_orca_format_prompt


################################################################################
# start of training
################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='')
    args = parser.parse_args()
    # Output directory where the model predictions and checkpoints will be stored
    output_dir = args.output_dir
    # base_model = "microsoft/Phi-3-mini-4k-instruct"
    base_model = "mistralai/Mistral-7B-v0.1"

    data = DatasetsHandler(test=False, train=True, dev=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,  # Activates 4-bit precision loading
        bnb_4bit_quant_type=bnb_4bit_quant_type,  # nf4
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,  # float16
        bnb_4bit_use_double_quant=use_nested_quant,  # False
    )

    model, tokenizer = get_model_and_tokenizer(base_model, bnb_config)

    # Set training parameters
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
    )

    # Load the dataset
    print("Loading dataset")

    prompt_format_fn = get_prompt_formatter(base_model)

    train_prompts = [{'text': prompt_format_fn(data.train_dataset.pairs[i], data.train_dataset.definitions), 'label': data.train_dataset.labels[i]} for i in range(len(data.train_dataset))]
    val_prompts = [{'text': prompt_format_fn(data.dev_dataset.pairs[i], data.dev_dataset.definitions), 'label': data.dev_dataset.labels[i]} for i in range(len(data.dev_dataset))]

    # tolkenize the dataset
    print("Tokenizing dataset")

    def preprocessing_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=max_seq_length)

    train_dataset = Dataset.from_list(train_prompts)
    val_dataset = Dataset.from_list(val_prompts)

    tokenized_train_dataset = train_dataset.map(preprocessing_function, batched=True, remove_columns=['text'])
    tokenized_train_dataset.set_format("torch")

    tokenized_val_dataset = val_dataset.map(preprocessing_function, batched=True, remove_columns=['text'])
    tokenized_val_dataset.set_format("torch")

    # Data collator for padding a batch of examples to the maximum length seen in the batch
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Set supervised fine-tuning parameters
    trainer = WeightedCELossTrainer(
        model=model,
        args=training_arguments,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # trainer.train(resume_from_checkpoint='/cs/labs/tomhope/forer11/SciCo_Retrivel/mistral_v2_sfttrainer/no_def/model/checkpoint-10')
    trainer.train()
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)