from openai import OpenAI
import pickle
import re
import os
from enum import Enum
import json


class DataType(Enum):
    TRAIN = 'train'
    TEST = 'test'
    DEV = 'dev'


############ PARAMS ############
BATCH_SIZE = 1000
client = OpenAI()
MAX_TOKENS = 200
MODEL = "gpt-4o"
TERMS_PATH = '/cs/labs/tomhope/forer11/SciCo_Retrivel/definition_handler/data/full_texts/'
SAVE_PATH = '/cs/labs/tomhope/forer11/SciCo_Retrivel/definition_handler/data/gpt_4_definitions/batches_files/'


################################

def extract_term(text):
    return re.search(r'<m>(.*?)</m>', text).group(1)


def create_prompt(term_context):
    term = extract_term(term_context)
    return f'Provide a short definition for the term: {term} inside the following context: {term_context}'


def create_batch_line(line_number, term_context):
    batch_line = {"custom_id": f'{line_number}',
                  "method": "POST",
                  "url": "/v1/chat/completions",
                  "body": {"model": MODEL,
                           "messages": [{"role": "system",
                                         "content": "You will provide a short definition for the user's term"},
                                        {"role": "user",
                                         "content": create_prompt(term_context)}],
                           "max_tokens": MAX_TOKENS}}
    return batch_line


def create_batch_jsonl(save_path, batch, batch_number, batch_size, type='train'):
    batch_lines = [create_batch_line(i + (batch_number * batch_size), term_context) for i, term_context in
                   enumerate(batch)]
    with open(save_path + f'{type}_batch_{batch_number}.jsonl', 'w') as file:
        for line in batch_lines:
            file.write(json.dumps(line) + "\n")


def create_batches_json(terms_path, save_path, batch_size, type):
    with open(terms_path + f'{type}_terms_definitions_final.pickle', 'rb') as original_defs_file:
        terms_prompt_dict = pickle.load(original_defs_file)
    terms_context = sorted(list(terms_prompt_dict.keys()))
    terms_contexts_batches = [terms_context[i:i + batch_size] for i in range(0, len(terms_context), batch_size)]
    for i, batch in enumerate(terms_contexts_batches):
        create_batch_jsonl(save_path, batch, i, batch_size, type)
        print(f'Batch {i} created')


def get_batch_file_to_batch_id_dict(type, is_first_batch):
    if is_first_batch:
        print(f'Starting new batch file to id dict for {type}')
        return {}
    with open(SAVE_PATH + f'batch_file_to_batch_id_{type}_cont.pickle', 'rb') as file:
        batch_file_to_batch_id = pickle.load(file)
    return batch_file_to_batch_id


def send_batch_to_openai(batch_file, type, batch_index):
    batch_file_to_batch_id = get_batch_file_to_batch_id_dict(type, batch_index == 0)
    batch_input_file = client.files.create(
        file=open(SAVE_PATH + batch_file, "rb"),
        purpose="batch"
    )
    batch_input_file_id = batch_input_file.id

    batch = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": batch_file,
            "batch_id": str(batch_index)
        }
    )
    batch_file_to_batch_id[batch_file] = batch.id
    with open(SAVE_PATH + f'batch_file_to_batch_id_{type}_cont.pickle', 'wb') as file:
        pickle.dump(batch_file_to_batch_id, file)


def send_batches_to_openai(batch_files, type):
    for i, batch_file in enumerate(batch_files):
        send_batch_to_openai(batch_file, type, i)
        print(f'Sent {type} batch {i}')


def filter_files_by_type(file_list, type):
    return [file_name for file_name in file_list if file_name.startswith(type)]


def get_batch_files_by_data_type(path, type):
    try:
        # Get the list of files in the directory
        file_names = os.listdir(path)
        if not file_names:
            print("The directory is empty.")
        else:
            file_names = filter_files_by_type(file_names, type)
            sorted_files_names = sorted(file_names, key=lambda x: int(re.search(r'batch_(\d+)\.jsonl', x).group(1)))
            return filter_files_by_type(sorted_files_names, type)

    except FileNotFoundError:
        print("The directory does not exist.")
    except PermissionError:
        print("You do not have permission to access this directory.")


def get_batches_results(batches_results_path, terms_path, type, output_path):
    with open(terms_path + f'{type}_terms_definitions_final.pickle', 'rb') as original_defs_file:
        terms_prompt_dict = pickle.load(original_defs_file)
    terms_context = sorted(list(terms_prompt_dict.keys()))
    custom_id_to_definition = {}
    file_names = os.listdir(batches_results_path)
    if not file_names:
        print("The directory is empty.")
    else:
        for batch_jsonl_output in file_names:
            with open(batches_results_path + batch_jsonl_output, 'r', encoding='utf-8') as file:
                for line in file:
                    json_obj = json.loads(line.strip())
                    custom_id, definition = json_obj['custom_id'], json_obj['response']['body']['choices'][0]['message']['content']
                    custom_id_to_definition[custom_id] = definition
    gpt_def_dict = {}
    sorted_defs = sorted(custom_id_to_definition.items(), key=lambda item: int(item[0]))
    for i, (custom_id, definition) in enumerate(sorted_defs):
        gpt_def_dict[terms_context[i]] = definition
    with open (output_path + f'{type}_terms_definitions_{MODEL}_final.pickle', 'wb') as file:
        pickle.dump(gpt_def_dict, file)
    print(f'Done {type} definitions')



if __name__ == '__main__':
    data_type = DataType.TRAIN.value
    create_batches_json(TERMS_PATH, SAVE_PATH, BATCH_SIZE, data_type)
    batch_files = get_batch_files_by_data_type(SAVE_PATH, data_type)
    send_batches_to_openai(batch_files, data_type)


    batch_output_path = f'/cs/labs/tomhope/forer11/SciCo_Retrivel/definition_handler/data/gpt_4_definitions/{data_type}_batch_outputs/'
    def_files_path = '/cs/labs/tomhope/forer11/SciCo_Retrivel/definition_handler/data/gpt_4_definitions/def_files/'
    get_batches_results(batch_output_path, TERMS_PATH, data_type, def_files_path)
    print('Done')
