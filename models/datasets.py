import collections
import jsonlines
import numpy as np
import re
import torch
from itertools import product, combinations
from torch.utils import data
import pickle

class CrossEncoderDataset(data.Dataset):
    def __init__(self, data_path,
                 full_doc=True,
                 multiclass='multiclass',
                 sep_token='</s>',
                 is_training=True,
                 cdlm=False,
                 data_label='train',
                 only_hard_10=False,
                 should_save_term_context=False):
        super(CrossEncoderDataset, self).__init__()
        if full_doc:
            print('Using full doc')
        else:
            raise Exception("Use full doc only")
        self.should_save_term_context = should_save_term_context
        with jsonlines.open(data_path, 'r') as f:
            if only_hard_10:
                self.data = [topic for topic in f if topic['hard_10']]
            else:
                self.data = [topic for topic in f]

        # self.data = self.data[:2]
        # self.data = [self.data[150]]

        for i, topic in enumerate(self.data):
            self.data[i]['mention_text'] = np.array([' '.join(topic['flatten_tokens'][start:end + 1])
                                                     for start, end, _ in topic['flatten_mentions']])

        self.sep = sep_token
        self.cdlm = cdlm
        self.full_doc = full_doc
        if multiclass not in {'coref', 'hypernym', 'multiclass'}:
            raise ValueError(f"The multiclass value needs to be in (coref, hypernym, multiclass), got {multiclass}.")
        self.multiclass = multiclass
        self.is_training = is_training

        self.pairs, self.labels = [], []
        self.first, self.second = [], []
        self.info_pairs = []
        self.definitions = {}
        self.term_context_dict = {}

        print(f'Loading definitions from {data_label}')
        if self.full_doc:
            with open(f'/data/{data_label}_terms_definitions_final.pickle', "rb") as fp:
                self.definitions = pickle.load(fp)
        for i, topic in enumerate(self.data):
            if self.multiclass == 'multiclass':
                inputs, labels, info_pairs, definitions = self.get_topic_pairs(topic)
            elif self.multiclass == 'hypernym':
                inputs, labels, info_pairs = self.get_topic_pair_for_hypernym(topic)
            else:
                inputs, labels, info_pairs = self.get_topic_pairs_for_binary_classification(topic)

            self.pairs.extend(inputs)
            self.labels.extend(labels)
            pair_nums = len(info_pairs)
            info_pairs = np.concatenate((np.array([i] * pair_nums).reshape(pair_nums, 1),
                                         info_pairs), axis=1)
            self.info_pairs.extend(info_pairs)
        self.natural_labels = [str(x) for x in self.labels]

        if self.multiclass == 'multiclass' or self.multiclass == 'hypernym':
            print('counting pairs len...')
            total_length = len(self.pairs)
            print(f'total pairs length: {total_length}')
            self.labels = torch.tensor(self.labels, dtype=torch.long)
        else:
            self.labels = torch.tensor(self.labels, dtype=torch.float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.pairs[idx], self.labels[idx]

    def get_topic_pairs_for_binary_classification(self, topic):
        mentions = []
        for mention in topic['mentions']:
            if self.full_doc:
                mentions.append(self.get_full_doc_mention(mention, topic['tokens']))
            else:
                mentions.append(self.get_sentence_context(mention, topic['tokens'], topic['sentences']))
        mentions = np.array(mentions)

        if self.is_training:
            first, second = zip(*combinations(range(len(mentions)), r=2))
        else:
            first, second = zip(*product(range(len(mentions)), repeat=2))

        first, second = np.array(first), np.array(second)

        seps = np.array([self.sep] * len(first))
        inputs = np.char.add(np.char.add(mentions[first], seps), mentions[second]).tolist()

        labels = [topic['mentions'][x][-1] == topic['mentions'][y][-1]
                  for x, y in zip(first, second)]

        return inputs, labels, list(zip(first, second))

    def get_topic_pair_for_hypernym(self, topic):
        '''
                :param topic:
                :return:
                '''
        relations = [(x, y) for x, y in topic['relations']]
        mentions = []
        for mention in topic['mentions']:
            if self.full_doc:
                mentions.append(self.get_full_doc_mention(mention, topic['tokens']))
            else:
                mentions.append(self.get_sentence_context(mention, topic['tokens'], topic['sentences']))
        mentions = np.array(mentions)

        first, second = zip(*[(x, y) for x, y in product(range(len(mentions)), repeat=2) if x != y])
        first, second = np.array(first), np.array(second)

        seps = np.array([self.sep] * len(first))
        inputs = np.char.add(np.char.add(mentions[first], seps), mentions[second]).tolist()

        labels = []
        for x, y in zip(first, second):
            cluster_x, cluster_y = topic['mentions'][x][-1], topic['mentions'][y][-1]
            if (cluster_x, cluster_y) in relations:
                labels.append(1)
            elif (cluster_y, cluster_x) in relations:
                labels.append(2)
            else:
                labels.append(0)

        return inputs, labels, list(zip(first, second))

    def get_mentions_definitions(self, mentions):
        definitions = {}
        for mention_sentence in mentions:
            mention = re.search(r'<m>(.*?)</m>', mention_sentence).group(1).strip()
            definition = get_definition(self.definition_extraction_model, mention, mention_sentence)
            definitions[mention_sentence] = definition
        return definitions

    def create_mentions_context_dict(self, mentions):
        for mention_sentence in mentions:
            mention = re.search(r'<m>(.*?)</m>', mention_sentence).group(1).strip()
            self.term_context_dict[(mention, mention_sentence)] = mention_sentence

    def get_topic_pairs(self, topic):
        '''
        :param topic:
        :return:
        '''
        relations = [(x, y) for x, y in topic['relations']]
        mentions = []
        definitions = {}
        for mention in topic['mentions']:
            if self.full_doc:
                mentions.append(self.get_full_doc_mention(mention, topic['tokens']))
            else:
                mentions.append(self.get_sentence_context(mention, topic['tokens'], topic['sentences']))
        mentions = np.array(mentions)

        if self.should_save_term_context:
            self.create_mentions_context_dict(mentions)

        first, second = zip(*[(x, y) for x, y in product(range(len(mentions)), repeat=2) if x != y])
        first, second = np.array(first), np.array(second)

        inputs = np.char.add(mentions[first], mentions[second]).tolist()

        labels = []
        for x, y in zip(first, second):
            cluster_x, cluster_y = topic['mentions'][x][-1], topic['mentions'][y][-1]
            if cluster_x == cluster_y:
                labels.append(1)
            elif (cluster_x, cluster_y) in relations:
                labels.append(2)
            elif (cluster_y, cluster_x) in relations:
                labels.append(3)
            else:
                labels.append(0)

        return inputs, labels, list(zip(first, second)), definitions

    def get_full_doc_mention(self, mention, tokens):
        doc_id, start, end, _ = mention

        mention_rep = tokens[doc_id][:start] + ['<m>']
        mention_rep += tokens[doc_id][start:end + 1] + ['</m>']
        mention_rep += tokens[doc_id][end + 1:] + [self.sep]

        if self.cdlm:
            mention_rep = ['<doc-s>'] + mention_rep + ['</doc-s>']

        return ' '.join(mention_rep)

    def get_sentence_context(self, mention, tokens, sentences):
        doc_id, start, end, _ = mention
        sent_start, sent_end = 0, len(tokens) - 1
        i = 0
        while i < len(sentences[doc_id]):
            sent_start, sent_end = sentences[doc_id][i]
            if start >= sent_start and end <= sent_end:
                break
            i += 1

        mention_rep = tokens[doc_id][sent_start:start] + ['<m>']
        mention_rep += tokens[doc_id][start:end + 1] + ['</m>']
        mention_rep_with_sep = mention_rep + tokens[doc_id][end + 1:sent_end] + [self.sep]

        if self.cdlm:
            mention_rep = ['<doc-s>'] + mention_rep_with_sep + ['</doc-s>']

        return ' '.join(mention_rep_with_sep)


class BiEncoderDataset(data.Dataset):
    def __init__(self, data_path, full_doc=True, multiclass='multiclass', sep_token='</s>', is_training=True):
        super(BiEncoderDataset, self).__init__()
        with jsonlines.open(data_path, 'r') as f:
            self.data = [topic for topic in f]

        for i, topic in enumerate(self.data):
            self.data[i]['mention_text'] = np.array([' '.join(topic['flatten_tokens'][start:end + 1])
                                                     for start, end, _ in topic['flatten_mentions']])

        self.sep = sep_token
        self.full_doc = full_doc
        if multiclass not in {'coref', 'hypernym', 'multiclass'}:
            raise ValueError(f"The multiclass value needs to be in (coref, hypernym, multiclass), got {multiclass}.")
        self.multiclass = multiclass
        self.is_training = is_training

        self.pairs, self.labels = [], []
        self.first, self.second = [], []
        self.info_pairs = []
        for i, topic in enumerate(self.data):
            if self.multiclass == 'multiclass':
                m1, m2, labels, info_pairs = self.get_topic_pairs(topic)
            elif self.multiclass == 'hypernym':
                m1, m2, labels, info_pairs = self.get_topic_pair_for_hypernym(topic)
            else:
                m1, m2, labels, info_pairs = self.get_topic_pairs_for_binary_classification(topic)

            self.first.extend(m1)
            self.second.extend(m2)

            # self.pairs.extend(inputs)
            self.labels.extend(labels)
            pair_nums = len(info_pairs)
            info_pairs = np.concatenate((np.array([i] * pair_nums).reshape(pair_nums, 1),
                                         info_pairs), axis=1)
            self.info_pairs.extend(info_pairs)

        if self.multiclass == 'multiclass' or self.multiclass == 'hypernym':
            self.labels = torch.tensor(self.labels, dtype=torch.long)
        else:
            self.labels = torch.tensor(self.labels, dtype=torch.float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.first[idx], self.second[idx], self.labels[idx]

    def get_topic_pairs_for_binary_classification(self, topic):
        mentions = []
        for mention in topic['mentions']:
            if self.full_doc:
                mentions.append(self.get_full_doc_mention(mention, topic['tokens']))
            else:
                mentions.append(self.get_sentence_context(mention, topic['tokens'], topic['sentences']))
        mentions = np.array(mentions)

        if self.is_training:
            first, second = zip(*combinations(range(len(mentions)), r=2))
        else:
            first, second = zip(*product(range(len(mentions)), repeat=2))

        first, second = np.array(first), np.array(second)

        m1 = mentions[first]
        m2 = mentions[second]
        #
        # seps = np.array([self.sep] * len(first))
        # inputs = np.char.add(np.char.add(mentions[first], seps), mentions[second]).tolist()

        labels = [topic['mentions'][x][-1] == topic['mentions'][y][-1]
                  for x, y in zip(first, second)]

        return m1, m2, labels, list(zip(first, second))

    def get_topic_pair_for_hypernym(self, topic):
        '''
                :param topic:
                :return:
                '''
        relations = [(x, y) for x, y in topic['relations']]
        mentions = []
        for mention in topic['mentions']:
            if self.full_doc:
                mentions.append(self.get_full_doc_mention(mention, topic['tokens']))
            else:
                mentions.append(self.get_sentence_context(mention, topic['tokens'], topic['sentences']))
        mentions = np.array(mentions)

        first, second = zip(*[(x, y) for x, y in product(range(len(mentions)), repeat=2) if x != y])
        first, second = np.array(first), np.array(second)

        m1 = mentions[first]
        m2 = mentions[second]
        # seps = np.array([self.sep] * len(first))
        # inputs = np.char.add(np.char.add(mentions[first], seps), mentions[second]).tolist()

        labels = []
        for x, y in zip(first, second):
            cluster_x, cluster_y = topic['mentions'][x][-1], topic['mentions'][y][-1]
            if (cluster_x, cluster_y) in relations:
                labels.append(1)
            elif (cluster_y, cluster_x) in relations:
                labels.append(2)
            else:
                labels.append(0)

        return m1, m2, labels, list(zip(first, second))

    def get_topic_pairs(self, topic):
        '''
        :param topic:
        :return:
        '''
        relations = [(x, y) for x, y in topic['relations']]
        mentions = []
        for mention in topic['mentions']:
            if self.full_doc:
                mentions.append(self.get_full_doc_mention(mention, topic['tokens']))
            else:
                mentions.append(self.get_sentence_context(mention, topic['tokens'], topic['sentences']))
        mentions = np.array(mentions)

        first, second = zip(*[(x, y) for x, y in product(range(len(mentions)), repeat=2) if x != y])
        first, second = np.array(first), np.array(second)

        m1 = mentions[first]
        m2 = mentions[second]
        # seps = np.array([self.sep] * len(first))
        # inputs = np.char.add(np.char.add(mentions[first], seps), mentions[second]).tolist()

        labels = []
        for x, y in zip(first, second):
            cluster_x, cluster_y = topic['mentions'][x][-1], topic['mentions'][y][-1]
            if cluster_x == cluster_y:
                labels.append(1)
            elif (cluster_x, cluster_y) in relations:
                labels.append(2)
            elif (cluster_y, cluster_x) in relations:
                labels.append(3)
            else:
                labels.append(0)

        return m1, m2, labels, list(zip(first, second))

    def get_full_doc_mention(self, mention, tokens):
        doc_id, start, end, _ = mention
        mention_rep = tokens[doc_id][:start] + ['<m>']
        mention_rep += tokens[doc_id][start:end + 1] + ['</m>']
        mention_rep += tokens[doc_id][end + 1:]
        return ' '.join(mention_rep)

    def get_sentence_context(self, mention, tokens, sentences):
        doc_id, start, end, _ = mention
        sent_start, sent_end = 0, len(tokens) - 1
        i = 0
        while i < len(sentences[doc_id]):
            sent_start, sent_end = sentences[doc_id][i]
            if start >= sent_start and end <= sent_end:
                break
            i += 1

        mention_rep = tokens[doc_id][sent_start:start] + ['<m>']
        mention_rep += tokens[doc_id][start:end + 1] + ['</m>']
        mention_rep += tokens[doc_id][end + 1:sent_end] + [self.sep]

        return ' '.join(mention_rep)


class ClusterForHypernymDataset(data.Dataset):
    '''
    generate all pairs of clusters (product)
    where each cluster is the concatenation of the all the mentions
    in the document
    labels:
    0: not related
    1: neutral == hypernym
    2: entailment == hyponym
    '''

    def __init__(self, data_path, with_context=False):
        super(ClusterForHypernymDataset, self).__init__()
        with jsonlines.open(data_path, 'r') as f:
            self.data = [topic for topic in f]
        self.with_context = with_context
        self.premise = []
        self.hypothesis = []
        self.labels = []

        for i, topic in enumerate(self.data):
            self.data[i]['mention_text'] = np.array([' '.join(topic['flatten_tokens'][start:end + 1])
                                                     for start, end, _ in topic['flatten_mentions']])

        for topic_num, topic in enumerate(self.data):
            premise, hypothesis, labels = self.get_topic_candidates(topic)
            self.premise.extend(premise)
            self.hypothesis.extend(hypothesis)
            self.labels.extend(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.premise[idx], self.hypothesis[idx], self.labels[idx]

    def get_topic_candidates(self, topic):
        clusters = collections.defaultdict(list)
        for i, (_, _, _, c_id) in enumerate(topic['mentions']):
            clusters[c_id].append(i)

        relations = [(x, y) for x, y in topic['relations']]
        cluster_ids, candidates = [], []
        for c_id, mentions in clusters.items():
            mention_text = topic['mention_text'][mentions]
            sampled = np.random.choice(mention_text, min(len(mention_text), 10), replace=False)
            candidates.append(', '.join(sampled))
            cluster_ids.append(c_id)

        permutations = [(x, y) for x, y in product(range(len(candidates)), repeat=2) if x != y]
        labels = []
        for x, y in permutations:
            if (x, y) in relations:
                labels.append(1)
            elif (y, x) in relations:
                labels.append(2)
            else:
                labels.append(0)
        first, second = zip(*permutations)
        first, second = torch.tensor(first), torch.tensor(second)

        candidates = np.array(candidates)
        premise = candidates[first]
        hypothesis = candidates[second]

        return premise, hypothesis, labels
