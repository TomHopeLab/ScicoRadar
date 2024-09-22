import argparse
import torch
from transformers import Phi3ForSequenceClassification, AutoTokenizer, pipeline, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import PeftModel
from definition_handler.process_data import DatasetsHandler
import re
from tqdm import tqdm
import pickle
from accelerate import Accelerator
from accelerate.utils import gather_object

# base_model = "microsoft/Phi-3-mini-4k-instruct"
base_model = "mistralai/Mistral-7B-v0.1"
max_seq_length = 1664

# device_map = 'auto'
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
# prompts format
################################################################################

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
def get_orca_format_prompt(pair, with_def=False, def_dict = None, relational_def = {}):
    term1_text, term2_text, _ = pair.split('</s>')
    term1 = re.search(r'<m>(.*?)</m>', term1_text).group(1).strip()
    term2 = re.search(r'<m>(.*?)</m>', term2_text).group(1).strip()
    term1_text, term2_text = term1_text.replace('<m> ', '').replace(' </m>', ''), term2_text.replace('<m> ',
                                                                                                     '').replace(
        ' </m>', '')

    text1, text2 = pair.split('</s>')[0] + '</s>', pair.split('</s>')[1] + '</s>'
    reversed_pair = text2 + text1
    term1_def, term2_def = def_dict[text1], def_dict[text2]
    if pair in relational_def:
        term1_def = relational_def[pair]
    if reversed_pair in relational_def:
        term2_def = relational_def[reversed_pair]
    return orca_template_with_def.format(term1=term1, term1_text=term1_text, term2=term2,
                                         term2_text=term2_text, term1_def=term1_def, term2_def=term2_def)


def get_phi3_instruct_prompt(pair, with_def=False, def_dict=None):
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

def get_phi3_model_and_tokenizer(base_model, bnb_config):
    model = Phi3ForSequenceClassification.from_pretrained(
        base_model,
        # quantization_config=bnb_config,
        device_map=device_map,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        num_labels=4,
        torch_dtype=torch.float16,
        # use_cache = False
    )

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
            # quantization_config=bnb_config,
            device_map=device_map,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
            num_labels=4,
            torch_dtype=torch.float16,
            # use_cache = False
        )

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_relational_def', type=str, default='False')
    parser.add_argument('--adapter', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='')
    args = parser.parse_args()

    use_relational_def = True if args.use_relational_def == 'True' else False
    adapter = args.adapter
    output_dir = args.output_dir

    data = DatasetsHandler(test=True, train=False, dev=False)

    accelerator = Accelerator()
    device_map = {"": accelerator.process_index}

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,  # Activates 4-bit precision loading
        bnb_4bit_quant_type=bnb_4bit_quant_type,  # nf4
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,  # float16
        bnb_4bit_use_double_quant=use_nested_quant,  # False
    )

    model, tokenizer = get_model_and_tokenizer(base_model, bnb_config)
    # Load and activate the adapter on top of the base model
    model = PeftModel.from_pretrained(model, adapter)

    # Merge the adapter with the base model
    model = model.merge_and_unload()


    prompt_format_fn = get_prompt_formatter(base_model)

    relational_def = {}
    if use_relational_def:
        with open(f'/data/test_relational_definitions.pickle', 'rb') as file:
            relational_def = pickle.load(file)

    test_prompts = [{'text': prompt_format_fn(data.test_dataset.pairs[i], data.test_dataset.definitions, relational_def),
                     'label': data.test_dataset.labels[i], "pair": data.test_dataset.pairs[i]} for i in range(len(data.test_dataset.pairs))]
    # results, test_prompts = combine_results_and_get_remaining_data(test_prompts)
    # sync GPUs and start the timer
    accelerator.wait_for_everyone()

    # divide the prompt list onto the available GPUs
    with accelerator.split_between_processes(test_prompts) as prompts:
        results = {}
        with torch.no_grad():
            for i, example in enumerate(tqdm(prompts, disable=not accelerator.is_local_main_process)):
                input = tokenizer(example['text'], return_tensors="pt", truncation=True, padding=True,
                                  max_length=max_seq_length).to('cuda')
                output = model.forward(**input).logits
                text, label, pair = example['text'], example['label'], example['pair']
                results[pair] = output

            results = [results]  # transform to list, otherwise gather_object() will not collect correctly

    results_gathered = gather_object(results)

    if accelerator.is_main_process:
        print(f'Processed all examples')
        with open(f'{output_dir}/final_results.pickle', 'wb') as file:
            pickle.dump(results_gathered, file)
        merged_results = {k: v for d in results_gathered for k, v in d.items()}
        with open(f'{output_dir}/merged_final_results.pickle', 'wb') as file:
            pickle.dump(merged_results, file)


# check https://github.com/ultralytics/ultralytics/issues/1439 and also look here https://github.com/huggingface/accelerate/issues/314