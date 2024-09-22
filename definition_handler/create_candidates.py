from process_data import DatasetsHandler
import pickle
import torch
import enum


class PairsProbs(enum.IntEnum):
    Prob = 0
    Label = 1


prompts_save_directory = '/cs/labs/tomhope/forer11/SciCo_Retrivel/definition_handler/data/full_texts'

logits_save_directory = '/cs/labs/tomhope/forer11/SciCo_Retrivel/mistral_1_classification/with_def/results/merged_final_results.pickle'



def build_first_sentence_map(pairs, labels):
    first_sentence_map = {}
    for idx, pair in enumerate(pairs):
        first_sentence = pair.split('</s>')[0] + '</s>'
        if first_sentence not in first_sentence_map:
            first_sentence_map[first_sentence] = []
        # if the label is 0 (no relationship), we don't want to add it to the map
        if labels[pair][PairsProbs.Label] != 0:
            first_sentence_map[first_sentence].append(idx)
    return first_sentence_map


def sort_first_sentence_map(first_sentence_map, logits_probs, pairs):
    # Iterate through each key-value pair in first_sentence_map
    for sentence in first_sentence_map:
        # Fetch the array of indices
        indices = first_sentence_map[sentence]

        # Create a list of tuples (prob, index) for sorting
        prob_index_pairs = [
            (logits_probs[pairs[idx]][0], idx) for idx in indices
        ]

        # Sort the list of tuples based on the prob value
        sorted_pairs = sorted(prob_index_pairs, key=lambda x: x[0], reverse=True)

        # Extract the sorted indices
        sorted_indices = [pair[1] for pair in sorted_pairs]

        # Update the first_sentence_map with the sorted indices
        first_sentence_map[sentence] = sorted_indices

    return first_sentence_map


def create_candidates(datasets, data_type):
    print('loading terms')
    with open(f'{prompts_save_directory}/{data_type}_terms_prompt_dict.pickle', 'rb') as file:
        terms_prompt_dict = pickle.load(file)

    print('loading logits')
    with open(f'{logits_save_directory}_cpu', 'rb') as file:
        logits_dict = pickle.load(file)

    logits_probs = {x: (torch.max(torch.softmax(v, dim=1)).item(), torch.argmax(torch.softmax(v, dim=1)).item()) for x,v in logits_dict.items()}

    first_sentence_map = build_first_sentence_map(datasets.test_dataset.pairs, logits_probs)

    sorted_first_sentence_map = sort_first_sentence_map(first_sentence_map, logits_probs, datasets.test_dataset.pairs)

    non_empty_first_sentence_map = {x: v for x, v in sorted_first_sentence_map.items() if len(v) > 0}

    # replace first ABSTRACT with CONTEXT
    terms_prompt_dict = {key: value.replace("ABSTRACT:", "CONTEXT:", 1) for key, value in terms_prompt_dict.items()}
    terms_prompt_dict = {key: value.replace("ABSTRACT:", "PAPER SNIPPET:") for
                 key, value in terms_prompt_dict.items()}

    return non_empty_first_sentence_map, terms_prompt_dict
