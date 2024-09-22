import dspy
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShotWithRandomSearch, BayesianSignatureOptimizer
from definition_handler.process_data import DatasetsHandler
import random
import pickle

NUM_OF_TRAIN_DATA = 200
NUM_OF_DEV_DATA = 60
OPENAI_API_KEY = 'sk-proj-k9gRnfEJYDeAOYtdBQR6T3BlbkFJebbL7HAYQLwEZAag9JSG'


# TODO remove tags!!!! and add the terms as data

class SCICO(dspy.Signature):
    (
        """You are given 2 texts, each one is a context for a scientific concept"""
        """ You must decide the correct relationship between the two concepts from the next options
        1 - Co-referring terms: Both term1 and term2 refer to the same underlying concept or entity.
        2 - Parent concept: Term1 represents a broader category or concept that encompasses term2, such that mentioning term1 implicitly invokes term2.
        3 - Child concept: The inverse of a parent concept relation. Term1 is a specific instance or subset of the broader concept represented by term2, such that mentioning term2 implicitly invokes term1.
        0 - None of the above: Term1 and term2 are not co-referring, and do not have a parent-child or child-parent relation.""")

    text_1 = dspy.InputField()
    text_2 = dspy.InputField()
    answer = dspy.OutputField(
        desc="{0, 1, 2, 3}")


class ScicoWithDef(dspy.Signature):
    (
        """You are given 2 texts, each one is a context for a scientific concept, and a definition for each concept"""
        """You must decide the correct relationship between the two concepts from the next options
        1 - Co-referring terms: Both term1 and term2 refer to the same underlying concept or entity.
        2 - Parent concept: Term1 represents a broader category or concept that encompasses term2, such that mentioning term1 implicitly invokes term2.
        3 - Child concept: The inverse of a parent concept relation. Term1 is a specific instance or subset of the broader concept represented by term2, such that mentioning term2 implicitly invokes term1.
        0 - None of the above: Term1 and term2 are not co-referring, and do not have a parent-child or child-parent relation.""")

    text_1 = dspy.InputField()
    definition_1 = dspy.InputField(desc='a contextual definition for the first concept')
    text_2 = dspy.InputField()
    definition_2 = dspy.InputField(desc='a contextual definition for the second concept')
    answer = dspy.OutputField(
        desc="{0, 1, 2, 3.}")


class BaseSCICOModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_hierarchy = dspy.Predict(SCICO)

    def forward(self, text_1, text_2):
        return self.generate_hierarchy(text_1=text_1, text_2=text_2)


class CoTSCICOModule(dspy.Module):
    def __init__(self):
        super().__init__()

        self.generate_hierarchy = dspy.ChainOfThought(SCICO)

    def forward(self, text_1, text_2):
        return self.generate_hierarchy(text_1=text_1, text_2=text_2)


class CoTScicoWithDefModule(dspy.Module):
    def __init__(self):
        super().__init__()

        self.generate_hierarchy = dspy.ChainOfThought(ScicoWithDef)

    def forward(self, text_1, text_2, definition_1, definition_2):
        return self.generate_hierarchy(text_1=text_1, text_2=text_2, definition_1=definition_1,
                                       definition_2=definition_2)


def get_both_sentences(sentence):
    sentences = sentence.split('</s>')
    return sentences[0], sentences[1]


def get_definitions(pair, def_dict):
    sent_1, sent_2 = pair
    return def_dict[sent_1 + '</s>'], def_dict[sent_2 + '</s>']


def get_dspy_example(data_set, num_of_data, shuffle=True, all_data=False, with_def=False):
    if shuffle:
        random.seed(4)
        label_0_indices = random.sample([index for index, label in enumerate(data_set.natural_labels) if label == '0'],
                                        num_of_data // 4)
        label_1_indices = random.sample([index for index, label in enumerate(data_set.natural_labels) if label == '1'],
                                        num_of_data // 4)
        label_2_indices = random.sample([index for index, label in enumerate(data_set.natural_labels) if label == '2'],
                                        num_of_data // 4)
        label_3_indices = random.sample([index for index, label in enumerate(data_set.natural_labels) if label == '3'],
                                        num_of_data // 4)
        indexes = label_0_indices + label_1_indices + label_2_indices + label_3_indices
        random.shuffle(indexes)

    elif all_data:
        indexes = [i for i in range(len(data_set))]
    else:
        indexes = [i for i in range(0, num_of_data, 200)]

    texts = [(get_both_sentences(data_set.pairs[i])) for i in indexes]
    labels = [data_set.natural_labels[i] for i in indexes]

    if with_def:
        definitions = [get_definitions(sentences, data_set.definitions) for sentences in texts]
        ## TODO remove later
        return [
            dspy.Example(
                text_1=texts[i][0],
                text_2=texts[i][1],
                definition_1=definitions[i][0],
                definition_2=definitions[i][1],
                answer=labels[i])
            .with_inputs('text_1', 'text_2', 'definition_1', 'definition_2') for i in range(len(texts))
        ]

    return [
        dspy.Example(
            text_1=texts[i][0],
            text_2=texts[i][1],
            answer=labels[i])
        .with_inputs('text_1', 'text_2') for i in range(len(texts))
    ]


def save_scores(results_path, output_path, data):
    sentences_to_score_dict = {}
    #
    with open(results_path, "rb") as file:
        loaded_data3 = pickle.load(file)

    for i, sentences in enumerate(data.test_dataset.pairs):
        sentences_to_score_dict[sentences] = loaded_data3['answers'][i]

    with open(output_path, "wb") as file:
        pickle.dump(sentences_to_score_dict, file)

    print("Saved scores to ", output_path)



data = DatasetsHandler(test=True, train=True, dev=True, only_hard_10=True, full_doc=True, should_load_definition=True)
# save_scores("/cs/labs/tomhope/forer11/SciCo_Retrivel/DSPY/gpt4-mini/with_def_no_opt/v4/score_results_until_70000.pkl",
#             "/cs/labs/tomhope/forer11/SciCo_Retrivel/DSPY/gpt4-mini/with_def_no_opt/v4/sentences_to_score_dict.pkl",
#             data
#             )
# save_scores("/cs/labs/tomhope/forer11/SciCo_Retrivel/DSPY/gpt4-mini/with_def_no_opt/v5/score_results_until_70000.pkl",
#             "/cs/labs/tomhope/forer11/SciCo_Retrivel/DSPY/gpt4-mini/with_def_no_opt/v5/sentences_to_score_dict.pkl",
#             data
#             )
# save_scores("/cs/labs/tomhope/forer11/SciCo_Retrivel/DSPY/gpt4-mini/with_def/v3/score_results_until_70000.pkl",
#             "/cs/labs/tomhope/forer11/SciCo_Retrivel/DSPY/gpt4-mini/with_def/v3/sentences_to_score_dict.pkl",
#             data
#             )


train = get_dspy_example(data.train_dataset, NUM_OF_TRAIN_DATA, with_def=True)
dev = get_dspy_example(data.dev_dataset, NUM_OF_DEV_DATA, with_def=True)
test = get_dspy_example(data.test_dataset, len(data.test_dataset), shuffle=False, all_data=True, with_def=True)
test_1000 = get_dspy_example(data.test_dataset, 1000, shuffle=True, all_data=True, with_def=True)
# test_for_print_def, test_for_print = get_dspy_example(data.test_dataset, 20, shuffle=True, all_data=False,
#                                                       with_def=False)

print(
    f"For this dataset, training examples have input keys {train[0].inputs().keys()} and label keys {train[0].labels().keys()}")

# turbo = dspy.OpenAI(model='gpt-4o-mini', model_type='chat', max_tokens=1600, api_key=OPENAI_API_KEY)
turbo = dspy.OpenAI(model='gpt-4o-mini', max_tokens=16000, api_key=OPENAI_API_KEY, temperature=0)

# # GPT-4 will be used only to bootstrap CoT demos:
# gpt4T = dspy.OpenAI(model='gpt-4-0125-preview', max_tokens=350, model_type='chat', api_key=OPENAI_API_KEY)

accuracy = dspy.evaluate.metrics.answer_exact_match

dspy.settings.configure(lm=turbo)

fewshot_optimizer = BootstrapFewShotWithRandomSearch(
    max_bootstrapped_demos=7,
    max_labeled_demos=4,
    num_candidate_programs=7,
    num_threads=12,
    # teacher_settings=dict(lm=gpt4T),
    metric=accuracy)

cot_fewshot = CoTScicoWithDefModule()
# cot_fewshot = CoTSCICOModule()
# cot_fewshot = fewshot_optimizer.compile(cot_fewshot, trainset=train, valset=dev)
# cot_fewshot.save("/cs/labs/tomhope/forer11/SciCo_Retrivel/DSPY/gpt4_mini_gpt4_def_v5.json")

cot_fewshot.load("/cs/labs/tomhope/forer11/SciCo_Retrivel/DSPY/gpt4_mini_gpt4_def_no_opt_v1.json")

cot_fewshot(**test[32000].inputs())
print(turbo.inspect_history(n=1))

#
#
# evaluator = Evaluate(devset=test_1000, num_threads=4, display_progress=True, display_table=0, return_outputs=True)
# score, results = evaluator(cot_fewshot, metric=accuracy)
# print('yay')



# print("Starting evaluation for gpt4_mini_no_def_no_opt")
# chunk_size = 1000
# # with open("/cs/labs/tomhope/forer11/SciCo_Retrivel/DSPY/sorted_results/score_results_until_29000.pkl", "rb") as file:
# #     loaded_data = pickle.load(file)
# # all_answers = loaded_data['answers']
# all_answers = []
# all_results = []
# for i in range(0, len(test), chunk_size):
#     chunk = test[i:i + chunk_size]
#     print("Evaluating until: ", i + chunk_size)
#     is_success = False
#     while not is_success:
#         try:
#             evaluator = Evaluate(devset=chunk, num_threads=4, display_progress=True, display_table=0,
#                                  return_outputs=True)
#             score, results = evaluator(cot_fewshot, metric=accuracy)
#             answers = [prediction.answer for example, prediction, temp_score in results]
#             rationals = [prediction.completions._completions['rationale'][0] for example, prediction, temp_score in
#                          results]
#             all_answers.extend(answers)
#             all_results.extend(results)
#             is_success = True
#         except Exception as e:
#             print(e)
#             print("Retrying...")
#     with open(
#             f'/cs/labs/tomhope/forer11/SciCo_Retrivel/DSPY/gpt4-mini/no_def/v1/score_results_until_{i + chunk_size}.pkl',
#             "wb") as file:
#         pickle.dump({'score': score, 'answers': all_answers, 'rationals': rationals}, file)
#     print("Processed chunk", i // chunk_size)

# cot_fewshot(**test[0].inputs())
# print(turbo.inspect_history(n=1))


# cot_zeroshot = CoTSCICOModule()
# kwargs = dict(num_threads=8, display_progress=True, display_table=0)
# optuna_trials_num =10 # Use more trials for better results
# teleprompter = BayesianSignatureOptimizer(task_model=turbo, prompt_model=turbo, metric=accuracy, n=5, init_temperature=1.0, verbose=True)
# compiled_prompt_opt = teleprompter.compile(cot_zeroshot, devset=dev, optuna_trials_num=optuna_trials_num, max_bootstrapped_demos=4, max_labeled_demos=4, eval_kwargs=kwargs)
# compiled_prompt_opt.save("/cs/labs/tomhope/forer11/SciCo_Retrivel/DSPY/BayesianSignatureOptimizer_program_2.json")

# cot_fewshot(**test[1].inputs())
# print(turbo.inspect_history(n=1))

# cot_fewshot = CoTScicoWithDefModule()
# cot_fewshot.load("/cs/labs/tomhope/forer11/SciCo_Retrivel/DSPY/BayesianSignatureOptimizer_program_with_def_2.json")
# cot_fewshot = bootstrap_optimizer.compile(cot_fewshot, trainset=train, valset=dev)
# cot_fewshot.save("/cs/labs/tomhope/forer11/SciCo_Retrivel/DSPY/cot_def_new_with_sig_opt.json")

# cot_fewshot = CoTScicoWithDefModule()
# cot_fewshot.load("/cs/labs/tomhope/forer11/SciCo_Retrivel/DSPY/BayesianSignatureOptimizer_program_with_def_2.json")
# cot_fewshot(**test[0].inputs())
# print(turbo.inspect_history(n=1))


# evaluator = Evaluate(devset=test, num_threads=1, display_progress=True, display_table=0)
# # basic_module = BaseSCICOModule()
# # basic_module(**test[0].inputs())
# cot_module = CoTSCICOModule()
# cot_module(**test[0].inputs())
# print(turbo.inspect_history(n=1))


# cot_fewshot = CoTSCICOModule()
# cot_fewshot.load("/cs/labs/tomhope/forer11/SciCo_Retrivel/DSPY/BayesianSignatureOptimizer_program_2.json")
# cot_fewshot(**test[0].inputs())
# print(turbo.inspect_history(n=1))


## examples for prompts:
# cot_fewshot = CoTSCICOModule()
# cot_fewshot.load("/cs/labs/tomhope/forer11/SciCo_Retrivel/DSPY/BayesianSignatureOptimizer_program_2.json")
#
# cot_fewshot_with_def = CoTScicoWithDefModule()
# cot_fewshot_with_def.load(
#     "/cs/labs/tomhope/forer11/SciCo_Retrivel/DSPY/BayesianSignatureOptimizer_program_with_def_2.json")
#
# for i in range(20):
#     cot_fewshot(**test_for_print[i].inputs())
#     print('without def')
#     print(turbo.inspect_history(n=1))
#     cot_fewshot_with_def(**test_for_print_def[i].inputs())
#     print('with def')
#     print(turbo.inspect_history(n=1))
#     print('real prediction: ', test_for_print_def[i].labels()['answer'])
#     print('-------------------')
