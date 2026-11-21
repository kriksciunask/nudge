"""The non-Transformer models, shared by the notebooks and the tuning script (same code as in 02_baselines / 03_rnn)."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from nudge.train import VOCAB_SIZE


class LogisticRegression(nn.Module):
    """One linear layer + sigmoid on word counts: the perceptron of Week 3 s16."""

    def __init__(self, number_of_words):
        super().__init__()
        self.linear = nn.Linear(number_of_words, 1)

    def forward(self, features):
        return self.linear(features).squeeze(-1)


class RNNClassifier(nn.Module):
    """Embedding -> plain RNN, H_t = tanh(X_t W_xh + H_{t-1} W_hh + b_h) (Week 6 s28) -> H_T -> Linear -> one logit."""

    def __init__(self, vocab_size=VOCAB_SIZE, d=32, hidden=64, dropout=0.2):
        super().__init__()
        self.settings = {"vocab_size": vocab_size, "d": d, "hidden": hidden, "dropout": dropout}
        self.embedding = nn.Embedding(vocab_size, d, padding_idx=0)
        self.recurrent = nn.RNN(d, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden, 1)

    def forward(self, ids):
        lengths = (ids != 0).sum(dim=1).clamp(min=1).cpu()
        packed = pack_padded_sequence(self.embedding(ids), lengths, batch_first=True, enforce_sorted=False)
        _, last_hidden = self.recurrent(packed)
        return self.output(self.dropout(last_hidden[-1])).squeeze(-1)
