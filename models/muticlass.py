# from transformers import *
import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Any, Optional
import numpy as np
import torchmetrics as tm
import pickle

def get_gpt_score(inputs, labels, sentences_to_score):
    try:
        preds = [int(sentences_to_score[sent]) for sent in inputs]
    except:
        preds = [0 for _ in range(len(labels))]
    # chance = np.random.uniform(0, 1)
    # ind = np.random.randint(0, 4) if chance < 0.1 else label
    l = [[0, 0, 0, 0] for _ in range(len(inputs))]
    for i, pred in enumerate(preds):
        if pred < 0 or pred > 3:
            pred = 0
        l[i][pred] = 100
    return l

def get_classification_score(inputs, labels, sentences_to_score):
    # only works with batch size of 1 for now
    scores = sentences_to_score[inputs[0]]
    # batch_response = torch.cat([tensor.unsqueeze(0) for tensor in scores], dim=0)
    return scores


def get_global_attention(input_ids, start_token, end_token):
    global_attention_mask = torch.zeros(input_ids.shape)
    global_attention_mask[:, 0] = 1  # global attention to the CLS token
    start = torch.nonzero(input_ids == start_token)
    end = torch.nonzero(input_ids == end_token)
    globs = torch.cat((start, end))
    value = torch.ones(globs.shape[0])
    global_attention_mask.index_put_(tuple(globs.t()), value)
    return global_attention_mask


class MulticlassModel:
    def __init__(self):
        super(MulticlassModel, self).__init__()

    @classmethod
    def get_model(cls, config):
        return MulticlassCrossEncoderWithResults(config, num_classes=4)
class MulticlassCrossEncoderWithResults(pl.LightningModule):
    '''
    multiclass classification with labels:
    0 not related
    1 coref
    2. hypernym
    3. neutral (hyponym)
    '''

    def __init__(self, config, num_classes=4):
        super(MulticlassCrossEncoderWithResults, self).__init__()
        self.acc = tm.Accuracy(top_k=1, task="multiclass", num_classes=num_classes)
        self.f1 = tm.F1Score(task="multiclass", num_classes=num_classes, average='none')
        self.recall = tm.Recall(task="multiclass", num_classes=num_classes, average='none')
        self.val_precision = tm.Precision(task="multiclass", num_classes=num_classes, average='none')

        scores_path = config['scores_path']

        with open(scores_path ,"rb") as file:
            self.sentences_to_score = pickle.load(file)

    def forward(self, inputs, labels):
        # scores = get_gpt_score(inputs, labels, self.sentences_to_score)
        # scores = torch.tensor(scores, dtype=torch.float)

        scores = get_classification_score(inputs, labels, self.sentences_to_score)
        return scores

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def validation_epoch_end(self, outputs):
        self.log_metrics()

    def test_step(self, batch, batch_idx):
        pass

    def test_step_end(self, outputs):
        pass

    def test_epoch_end(self, outputs):
        pass

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None):
        x, y = batch
        y_hat = self(x, y)
        y_hat = torch.softmax(y_hat, dim=1)
        return y_hat

    def compute_metrics(self, y_hat, y):
        self.acc(y_hat, y)
        self.f1(y_hat, y)
        self.recall(y_hat, y)
        self.val_precision(y_hat, y)

    def log_metrics(self):
        self.log('acc', self.acc.compute())
        f1_negative, f1_coref, f1_hypernym, f1_hyponym = self.f1.compute()
        recall_negative, recall_coref, recall_hypernym, recall_hyponym = self.recall.compute()
        precision_negative, precision_coref, precision_hypernym, precision_hyponym = self.val_precision.compute()
        self.log('f1_coref', f1_coref)
        self.log('recall_coref', recall_coref)
        self.log('precision_coref', precision_coref)
        self.log('f1_hypernym', f1_hypernym)
        self.log('recall_hypernym', recall_hypernym)
        self.log('precision_hypernym', precision_hypernym)
        self.log('f1_hyponym', f1_hyponym)
        self.log('recall_hyponym', recall_hyponym)
        self.log('precision_hyponym', precision_hyponym)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.config['model']['lr'])

    def tokenize_batch(self, batch):
        inputs, labels = zip(*batch)
        labels = np.array(labels)

        return np.array(inputs), labels

