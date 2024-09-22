from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
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

instructor_persist_directory = '/cs/labs/tomhope/forer11/unarxive_instructor_embeddings/'
instructor_name = 'hkunlp/instructor-xl'

mxbai_persist_directory = '/cs/labs/tomhope/forer11/unarxive_chroma_gpu_mxbai'
mxbai_full_persist_directory = '/cs/labs/tomhope/forer11/unarxive_full_mxbai_chroma'
mxbai_name = 'mixedbread-ai/mxbai-embed-large-v1'

sfr_persist_directory = '/cs/labs/tomhope/forer11/unarxive_sfr_chroma'
sfr_name = 'Salesforce/SFR-Embedding-Mistral'

sys_msg = """You are a helpful AI assistant, you are an agent capable of reading and understanding scientific papers and defining scientific terms. here are the steps you should take to give a proper definition:

- Read abstracts: given scientific papers abstracts, please read them carefully with the user's query term in mind but do not mention them in the definition itself. the first abstract is the term context.
- Understand abstracts: after reading the abstracts, please understand them and try to extract the most important information from them regarding the user's query term, not all the information is relevant to the definition
- Generate definition: after reading the abstracts, please generate a short definition for the user's query term.

here is an EXAMPLE for a query and a required generated definition:

### START EXAMPLE ###

User: Please generate a short and concise definition for the term ML after reading the following abstracts:

ABSTRACT: example abstract 1

ABSTRACT: example abstract 2

ABSTRACT: example abstract 3

...

ABSTRACT: example abstract n

Assistant: definition: ML is a subset of AI where computers learn patterns from data to make predictions or decisions without explicit programming.

### END EXAMPLE ###

Let's get started. The users query is as follows:
"""


def combine_pickle_files_to_terms_definitions(pickle_paths, processed_abstracts):
    terms_definitions = {}
    for path in pickle_paths:
        with open(path, 'rb') as file:
            terms_definitions.update(pickle.load(file))
    return terms_definitions


def save_terms_definitions_from_pickle_to_json(pickle_paths, processed_abstracts):
    terms_definitions = combine_pickle_files_to_terms_definitions(pickle_paths, processed_abstracts)
    with open(
            '/cs/labs/tomhope/forer11/Retrieval-augmented-defenition-extractor/data/definitions_v2/v2_terms_definitions.json',
            'w') as file:
        json.dump(terms_definitions, file)


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

def instructions_query_format(abstracts_string, text):
    term = extract_term(text)
    query = f'Please generate a short and concise definition for the term {term} after reading the following abstracts:\n{abstracts_string}'
    return query


def instruction_format(sys_message: str, query: str):
    # note, don't "</s>" to the end
    return f'<s> [INST] {sys_message} [/INST]\nUser: {query}\nAssistant: definition: '

def get_missing_terms(terms_prompt_dict):
    with open(f'/cs/labs/tomhope/forer11/SciCo_Retrivel/definition_handler/data/train_terms_definitions_until_10800.pickle', 'rb') as file:
        terms_definitions = pickle.load(file)

    return {term: prompt for term, prompt in terms_prompt_dict.items() if term not in terms_definitions}, terms_definitions

def create_mentions_definitions_from_existing_docs_with_mistral_instruct(terms_dict, retriever_abstracts, retriever_all,
                                                                         data_type):
    print(f'creating terms_definitions with mistral_instruct for {data_type}...')
    # model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        # bnb_4bit_use_double_quant=True,
        # bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 cache_dir='/cs/labs/tomhope/forer11/cache',
                                                 attn_implementation="flash_attention_2",
                                                 trust_remote_code=True,
                                                 device_map="auto",
                                                 # quantization_config=bnb_config,
                                                 torch_dtype=torch.float16)
    generate_text = transformers.pipeline(
        model=model, tokenizer=tokenizer,
        device_map="auto",
        return_full_text=False,  # if using langchain set True
        task="text-generation",
        # we pass model parameters here too
        # temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
        # top_p=0.15,  # select from top tokens whose probability add up to 15%
        # top_k=0,  # select from top 0 tokens (because zero, relies on top_p)
        max_new_tokens=250,  # max number of tokens to generate in the output
        repetition_penalty=1.1  # if output begins repeating increase
    )
    generate_text.tokenizer.pad_token_id = model.config.eos_token_id
    terms_definitions = {}
    print('Processing Prompts...')
    if os.path.exists(
            f'/cs/labs/tomhope/forer11/SciCo_Retrivel/definition_handler/data/full_texts/{data_type}_terms_prompt_dict.pickle'):
        print('Loading terms_prompt_dict from pickle file...')
        with open(
                f'/cs/labs/tomhope/forer11/SciCo_Retrivel/definition_handler/data/full_texts/{data_type}_terms_prompt_dict.pickle',
                'rb') as file:
            terms_prompt_dict = pickle.load(file)
    else:
        reranker = CrossEncoder("mixedbread-ai/mxbai-rerank-large-v1")
        print('Creating terms_prompt_dict...')
        terms_prompt_dict = {}
        for term in tqdm(terms_dict):
            text = terms_dict[term]
            abstracts = get_abstracts_texts_formatted(term[0], text, retriever_abstracts, retriever_all, reranker)
            terms_prompt_dict[term[1]] = abstracts

        with open(
                f'/cs/labs/tomhope/forer11/SciCo_Retrivel/definition_handler/data/full_texts/{data_type}_terms_prompt_dict.pickle',
                'wb') as file:
            pickle.dump(terms_prompt_dict, file)

    # creat prompts from abstracts
    # query = instructions_query_format(abstracts, term)
    # prompt = instruction_format(sys_msg, query)
    terms_prompt_dict = {
        term: instruction_format(
            sys_msg, instructions_query_format(abstracts, term)
        ) for term, abstracts in terms_prompt_dict.items()
    }

    # terms_prompt_dict, terms_definitions = get_missing_terms(terms_prompt_dict)

    data = pd.DataFrame(list(terms_prompt_dict.items()), columns=['Term', 'Prompt'])
    dataset = Dataset.from_pandas(data)

    print('Generating definitions...')

    for i, out in tqdm(enumerate(generate_text(KeyDataset(dataset, 'Prompt'), batch_size=8)), total=len(dataset)):
        term = dataset[i]['Term']
        definition = out[0]['generated_text'].strip()
        terms_definitions[term] = definition
        if i % 100 == 0:
            print(f'Processed {i} terms')
            with open(
                    f'/cs/labs/tomhope/forer11/SciCo_Retrivel/definition_handler/data/full_texts/{data_type}_missing_terms_definitions_until_{i}.pickle',
                    'wb') as file:
                # Dump the dictionary into the file using pickle.dump()
                pickle.dump(terms_definitions, file)

    print('Saving terms_definitions to pickle file...')
    with open(
            f'/cs/labs/tomhope/forer11/SciCo_Retrivel/definition_handler/data/full_texts/{data_type}_terms_definitions_final.pickle',
            'wb') as file:
        # Dump the dictionary into the file using pickle.dump()
        pickle.dump(terms_definitions, file)


def embed_and_store(texts=[], load=True, persist_directory=instructor_persist_directory, hf_model_name='',
                    is_instructor=False):
    embedding = get_embeddings_model(hf_model_name, '/cs/labs/tomhope/forer11/cache/', is_instructor)

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


def process_abstracts_to_docs(abstracts):
    formatted_docs = []
    for doi in abstracts:
        # print(f'reading {root + file}...')
        page_content = abstracts[doi]
        # Create an instance of Document with content and metadata
        metadata = {'doi': doi}
        formatted_docs.append(Document(page_content=page_content, metadata=metadata))
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_documents(formatted_docs)
    return texts


def get_embeddings_model(embeddings_model_name, cache_folder, is_instructor):
    if is_instructor:
        return get_instructor_embeddings(embeddings_model_name, cache_folder)
    else:
        return get_huggingface_embeddings(embeddings_model_name, cache_folder)


def get_huggingface_embeddings(embeddings_model_name, cache_folder):
    return HuggingFaceEmbeddings(model_name=embeddings_model_name,
                                 cache_folder=cache_folder,
                                 model_kwargs={"device": "cuda"}
                                 # multi_process=True,
                                 # show_progress=True
                                 )


def get_instructor_embeddings(embeddings_model_name, cache_folder):
    # the default instruction is: 'Represent the document for retrieval:'
    return HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",
                                         model_kwargs={"device": "cuda"},
                                         cache_folder=cache_folder)


def process_arxive_to_docs():
    formatted_docs = []
    for root, dirs, files in os.walk("/cs/labs/tomhope/forer11/arXiv_data_handler/"):
        for file in files:
            print(f'reading {root + file}...')
            with open(root + file, "rb") as fp:
                docs = pickle.load(fp)
                for doc in docs:
                    page_content = doc['abstract']['text']
                    # Create an instance of Document with content and metadata
                    metadata = {key: value for key, value in doc['metadata'].items() if
                                isinstance(value, (str, int, float)) and key != 'abstract'}
                    metadata['discipline'] = doc['discipline']
                    # create a document for the abstract
                    formatted_docs.append(Document(page_content=page_content, metadata=metadata))
                    for body_text in doc['body_text']:
                        if len(body_text['text']) < 100:
                            continue
                        page_content = body_text['text']
                        # Create a Document for the body text
                        formatted_docs.append(Document(page_content=page_content, metadata=metadata))
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    texts = text_splitter.split_documents(formatted_docs)
    print(f'Processed {len(texts)} documents')
    return texts


def get_def_dict_from_json(json_path):
    with open(json_path, 'r') as file:
        terms_definitions = json.load(file)
    return terms_definitions


if __name__ == '__main__':
    datasets = DatasetsHandler(test=True, train=True, dev=True, full_doc=True)

    # print('creating docs...')
    # texts = process_arxive_to_docs()

    vector_store = embed_and_store([], True, mxbai_full_persist_directory, mxbai_name)
    retriever_all = vector_store.as_retriever(search_kwargs={"k": 12})
    vector_store = embed_and_store([], True, mxbai_persist_directory, mxbai_name)
    retriever_abstracts = vector_store.as_retriever(search_kwargs={"k": 12})

    create_mentions_definitions_from_existing_docs_with_mistral_instruct(datasets.train_dataset.term_context_dict, retriever_abstracts, retriever_all, 'train')

    # with open('/cs/labs/tomhope/forer11/SciCo_Retrivel/definition_handler/data/train_terms_definitions_final.pickle', 'rb') as file:
    #     yay = pickle.load(file)

    # terms_def = get_def_dict_from_json('/cs/labs/tomhope/forer11/Retrieval-augmented-defenition-extractor/data/definitions_v2/v2_terms_definitions.json')
    # print(len(terms_def))

    # print('searching...')
    # x = retriever.invoke('define the term MLP layer with this context: We apply dropout ( p = 0.5 ) on the output of the word embedding layer and the input and the output of the <m> MLP layer </m> .')
    # y = retriever.invoke('define the term numerical feature representation schemes with this context: Here , we present iFeature , a versatile Python‐based toolkit for generating various <m> numerical feature representation schemes </m> for both protein and peptide sequences .')
    # z = retriever.invoke('define the term text categorization problem with this context: Authorship attribution may be considered as a <m> text categorization problem </m> .')
    #
    # x2 = retriever2.invoke('define the term MLP layer with this context: We apply dropout ( p = 0.5 ) on the output of the word embedding layer and the input and the output of the <m> MLP layer </m> .')
    # y2 = retriever2.invoke('define the term numerical feature representation schemes with this context: Here , we present iFeature , a versatile Python‐based toolkit for generating various <m> numerical feature representation schemes </m> for both protein and peptide sequences .')
    # z2 = retriever2.invoke('define the term text categorization problem with this context: Authorship attribution may be considered as a <m> text categorization problem </m> .')

    print('Done!')
