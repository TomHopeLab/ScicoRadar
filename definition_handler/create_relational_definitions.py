from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import Chroma
from sentence_transformers import CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import transformers
import torch
from tqdm import tqdm
import pickle
import pandas as pd
from transformers.pipelines.pt_utils import KeyDataset
from datasets import Dataset
import json
import re
import os

from process_data import DatasetsHandler
from create_candidates import create_candidates

mxbai_persist_directory = '/cs/labs/tomhope/forer11/unarxive_chroma_gpu_mxbai'
mxbai_full_persist_directory = '/cs/labs/tomhope/forer11/unarxive_full_mxbai_chroma'
mxbai_name = 'mixedbread-ai/mxbai-embed-large-v1'

definitions_save_directory = '/cs/labs/tomhope/forer11/SciCo_Retrivel/definition_handler/data/relational_defibitions_full_mixtral'

sfr_persist_directory = '/cs/labs/tomhope/forer11/unarxive_sfr_chroma'
sfr_name = 'Salesforce/SFR-Embedding-Mistral'

# model_id = "mistralai/Mistral-7B-Instruct-v0.3"
model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
# model_id = "microsoft/Phi-3-medium-128k-instruct"
# model_id = 'MaziyarPanahi/Llama-3-70B-Instruct-DPO-v0.4'

process = 0

sys_msg = """You are a helpful AI assistant, you are an agent capable of reading and understanding scientific papers. 

you are capable of defining a scientific term with with respect to its relationship to another scientific term. here are the steps you should take to give a proper definition:

- Read the terms CONTEXT: under CONTEXT: for each term you will have the term's original context, meaning the context where the term was originally in. the context is important to understand the term better but sometimes it can be irrelevant to the term so keep it in mind.
- Read the PAPER SNIPPETS: given scientific papers snippets, please read them carefully with the user's query terms and contexts in mind but do not mention them in the definition itself.
- Understand the PAPER SNIPPETS: after reading the snippets, please understand them and try to extract the most important information from them regarding the user's query terms, not all the information is relevant to the definition and remember they were retrieved using the term and context.
- Generate definition: after reading the terms contexts and snippets, understand all the information fully and only then please generate a short definition ONLY for the term we want to define with respect to its relationship with the other term.

here is an EXAMPLE for a query and a required generated definition:

### START EXAMPLE ###

User: Please generate a short and concise definition ONLY for term ML with respect to its relationship to the term AI after reading the following contexts and snippets for each term:

TERM TO DEFINE: ML

CONTEXT: example context for the term ML

PAPER SNIPPET: example paper snippet 1

PAPER SNIPPET: example paper snippet 2

PAPER SNIPPET: example paper snippet 3

...

PAPER SNIPPET: example paper snippet n

TERM TO CHECK RELATIONSHIP WITH: AI

CONTEXT: example context for the term AI

PAPER SNIPPET: example paper snippet 1

PAPER SNIPPET: example paper snippet 2

PAPER SNIPPET: example paper snippet 3

...

PAPER SNIPPET: example paper snippet n

Assistant: 
Machine Learning (ML): A subset of Artificial Intelligence (AI) that enables systems to learn from data and improve their performance over time without being explicitly programmed for each task.

### END EXAMPLE ###

Let's get started. The users query is as follows:
"""

phi3_sys_msg = """You are a helpful AI assistant, you are an agent capable of reading and understanding scientific papers. 

you are capable of defining a scientific term with with respect to its relationship to another scientific term. here are the steps you should take to give a proper definition:

- Read the terms CONTEXT: under CONTEXT: for each term you will have the term's original context, meaning the context where the term was originally in. the context is important to understand the term better but sometimes it can be irrelevant to the term so keep it in mind.
- Read the PAPER SNIPPETS: given scientific papers snippets, please read them carefully with the user's query terms and contexts in mind but do not mention them in the definition itself.
- Understand the PAPER SNIPPETS: after reading the snippets, please understand them and try to extract the most important information from them regarding the user's query terms, not all the information is relevant to the definition and remember they were retrieved using the term and context.
- Generate definition: after reading the contexts and snippets, please generate a short definition ONLY for the term we want to define with respect to its relationship with the other term.

here is an EXAMPLE for a query and a required generated definition:

### START EXAMPLE ###

Please generate a short and concise definition ONLY for term ML with respect to its relationship to the term AI after reading the following contexts and snippets for each term:

TERM TO DEFINE: ML

CONTEXT: example context for the term ML

PAPER SNIPPET: example paper snippet 1

PAPER SNIPPET: example paper snippet 2

PAPER SNIPPET: example paper snippet 3

...

PAPER SNIPPET: example paper snippet n

TERM TO CHECK RELATIONSHIP WITH: AI

CONTEXT: example context for the term AI

PAPER SNIPPET: example paper snippet 1

PAPER SNIPPET: example paper snippet 2

PAPER SNIPPET: example paper snippet 3

...

PAPER SNIPPET: example paper snippet n

definition: Machine Learning (ML): A subset of Artificial Intelligence (AI) that enables systems to learn from data and improve their performance over time without being explicitly programmed for each task.

### END EXAMPLE ###

Let's get started. The users query is as follows:
"""

def get_retrieval_query(term, text):
    return f'define the term {term} with this context: {text}'


def get_abstracts_texts_formatted(term, text, retriever_abstracts, retriever_all, reranker):
    retrieval_query = get_retrieval_query(term, text)
    docs_from_all = retriever_all.invoke(retrieval_query)
    docs_from_all = [doc.page_content for doc in docs_from_all]
    docs_from_abstracts = retriever_abstracts.invoke(retrieval_query)
    docs_from_abstracts = [doc.page_content for doc in docs_from_abstracts]

    reranked_docs = reranker.rank(retrieval_query, docs_from_all + docs_from_abstracts, return_documents=True, top_k=5)
    abstracts = [text] + [doc['text'] for doc in reranked_docs]
    formatted_query = ''.join([f'ABSTRACT:\n{text}\n' for text in abstracts])
    return formatted_query


def extract_term(text):
    return re.search(r'<m>(.*?)</m>', text).group(1)


def instructions_query_format(term_to_define, term_to_define_snippets, term_with_relation, term_with_relation_snippets):
    query = (
        f'Please generate a short and concise definition ONLY for term {term_to_define} with respect to its relationship to the term {term_with_relation} '
        f'after reading the following contexts and snippets for each term:\nTERM TO DEFINE: {term_to_define}\n{term_to_define_snippets}'
        f'\nTERM TO CHECK RELATIONSHIP WITH: {term_with_relation}\n{term_with_relation_snippets}\n')
    return query


def instruction_format(sys_message: str, query: str):
    # note, don't "</s>" to the end
    return f'<s> [INST] {sys_message} [/INST]\nUser: {query}\nAssistant: definition: '

def phi3_format(sys_message: str, query: str):
    # note, don't "</s>" to the end
    return f'<|user|>\n{sys_message}\n\n{query}\n<|end|><|assistant|>\ndefinition: '


def get_missing_terms(splitted_terms, process):
    if process == 0:
        with open(
                f'/cs/labs/tomhope/forer11/SciCo_Retrivel/definition_handler/data/relational_defibitions_full_mixtral/test_missing_terms_definitions_until_63100_process0.pickle',
                'rb') as file:
            terms_definitions = pickle.load(file)
    elif process == 1:
        with open(
                f'/cs/labs/tomhope/forer11/SciCo_Retrivel/definition_handler/data/relational_defibitions_full_mixtral/test_missing_terms_definitions_until_47200_process1.pickle',
                'rb') as file:
            terms_definitions = pickle.load(file)
    else:
        raise Exception("process must be 0 or 1")

    return [(term, prompt) for term, prompt in splitted_terms if
            term not in terms_definitions], terms_definitions


def create_terms_prompts_dict(sorted_first_sentence_map, sentence_to_snippets, pairs):
    terms_prompts_dict = {}
    for sentence, indices in sorted_first_sentence_map.items():
        for idx in indices[:20]:
            pair = pairs[idx]
            sent1, sent2, _ = pair.split('</s>')
            sent1_snippets, sent2_snippets = sentence_to_snippets[sent1 + '</s>'], sentence_to_snippets[sent2 + '</s>']
            term1, term2 = extract_term(sent1), extract_term(sent2)
            if 'Phi-3' in model_id:
                query = instructions_query_format(term1, sent1_snippets, term2, sent2_snippets)
                terms_prompts_dict[pair] = phi3_format(phi3_sys_msg, query)
            else:
                query = instructions_query_format(term1, sent1_snippets, term2, sent2_snippets)
                terms_prompts_dict[pair] = instruction_format(sys_msg, query)
    return terms_prompts_dict


def create_relational_definitions(dataset, sorted_first_sentence_map, sentence_to_snippets, data_type):
    print(f'creating terms_definitions with mistral_instruct for {data_type}...')
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 cache_dir='/cs/labs/tomhope/forer11/cache',
                                                 attn_implementation="flash_attention_2",
                                                 trust_remote_code=True,
                                                 device_map="auto",
                                                 quantization_config=bnb_config,
                                                 torch_dtype=torch.bfloat16,
                                                 )
    generate_text = transformers.pipeline(
        model=model, tokenizer=tokenizer,
        device_map="auto",
        return_full_text=False,  # if using langchain set True
        task="text-generation",
        # we pass model parameters here too
        # temperature=0.2,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
        # top_p=0.3,  # select from top tokens whose probability add up to 15%
        # top_k=0,  # select from top 0 tokens (because zero, relies on top_p)
        max_new_tokens=200,  # max number of tokens to generate in the output
        repetition_penalty=1.1  # if output begins repeating increase
    )
    generate_text.tokenizer.pad_token_id = model.config.eos_token_id

    terms_prompt_dict = create_terms_prompts_dict(sorted_first_sentence_map, sentence_to_snippets, dataset.test_dataset.pairs)

    sorted_terms_prompt_dict = sorted(terms_prompt_dict.items())
    split_index = len(sorted_terms_prompt_dict) // 2
    process1 = sorted_terms_prompt_dict[:split_index]
    process2 = sorted_terms_prompt_dict[split_index:]

    if process == 0:
        print('process 0')
        splitted_terms = process1
        start_from = 63100
    elif process == 1:
        print('process 1')
        splitted_terms = process2
        start_from = 47200
    else:
        raise Exception("process must be 0 or 1")


    splitted_terms, terms_definitions = get_missing_terms(splitted_terms, process)

    data = pd.DataFrame(splitted_terms, columns=['Term', 'Prompt'])
    dataset = Dataset.from_pandas(data)

    print('Generating definitions...')

    # terms_definitions = {}

    for i, out in tqdm(enumerate(generate_text(KeyDataset(dataset, 'Prompt'), batch_size=8)), total=len(dataset)):
        term = dataset[i]['Term']
        definition = out[0]['generated_text'].strip()
        terms_definitions[term] = definition
        if i % 100 == 0:
            print(f'Processed {i} terms')
            with open(
                    f'{definitions_save_directory}/{data_type}_missing_terms_definitions_until_{start_from + i}_process{process}.pickle',
                    'wb') as file:
                # Dump the dictionary into the file using pickle.dump()
                pickle.dump(terms_definitions, file)

    print('Saving terms_definitions to pickle file...')
    with open(
            f'{definitions_save_directory}/{data_type}_terms_definitions_final_process{process}.pickle',
            'wb') as file:
        # Dump the dictionary into the file using pickle.dump()
        pickle.dump(terms_definitions, file)


def embed_and_store(texts=[], load=True, persist_directory='', hf_model_name=''):
    embedding = get_embeddings_model(hf_model_name, '/cs/labs/tomhope/forer11/cache/')

    if load:
        print(f'loading Vector embeddings from {persist_directory}...')
        # vectordb = FAISS.load_local(folder_path=persist_directory, embeddings=embedding, index_name="unarxive_index")
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    else:
        print(f'creating Vector embeddings to {persist_directory}...')
        vectordb = Chroma.from_documents(texts, embedding, persist_directory=persist_directory)
        # vectordb = FAISS.from_documents(documents=texts,
        #                                 embedding=embedding)
        # vectordb.save_local(folder_path=persist_directory, index_name='unarxive_index')
        print('Created Embeddings')
    return vectordb


def get_embeddings_model(embeddings_model_name, cache_folder):
    return get_huggingface_embeddings(embeddings_model_name, cache_folder)


def get_huggingface_embeddings(embeddings_model_name, cache_folder):
    return HuggingFaceEmbeddings(model_name=embeddings_model_name,
                                 cache_folder=cache_folder,
                                 model_kwargs={"device": "cuda"}
                                 # multi_process=True,
                                 # show_progress=True
                                 )


def get_def_dict_from_json(json_path):
    with open(json_path, 'r') as file:
        terms_definitions = json.load(file)
    return terms_definitions


if __name__ == '__main__':
    datasets = DatasetsHandler(test=True, train=False, dev=False, full_doc=True)

    sorted_first_sentence_map, sentence_to_snippets = create_candidates(datasets, 'test')

    # vector_store = embed_and_store([], True, mxbai_full_persist_directory, mxbai_name)
    # retriever_all = vector_store.as_retriever(search_kwargs={"k": 12})
    # vector_store = embed_and_store([], True, mxbai_persist_directory, mxbai_name)
    # retriever_abstracts = vector_store.as_retriever(search_kwargs={"k": 12})

    create_relational_definitions(datasets, sorted_first_sentence_map, sentence_to_snippets, 'test')

    print('Done!')
