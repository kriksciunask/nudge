"""The Transformer for Nudge, built from the course workshop (Workshop_Transformers_students/1_name_generator.ipynb).

`attention`, `SelfAttention` and `Block` are the workshop's code unchanged (cells 6 and 15).
What changes for Nudge (docs/course-map.md section 2):
1. a PADDING mask instead of the causal mask: a finished visit may look both ways, only <pad> is hidden (Week 8 s20);
2. one answer per visit: the output at the FIRST position (our first word is always "new"/"returning"),
   like the <cls> token of Week 8 s31;
3. one logit -> sigmoid (Week 3 s16), instead of a score for every next letter.
"""

import math

import torch
from torch import nn


def attention(q, k, v, mask):
    # q, k, v: (..., positions, d).  mask: True where a position may be looked at.   (workshop c6)
    d = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)          # softmax(QK^T / sqrt(d)) V, Week 8 s20
    scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(weights, v)
    return output, weights


class SelfAttention(nn.Module):
    # Multi-head self-attention (workshop c15, unchanged).

    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(d, d, bias=False)
        self.to_k = nn.Linear(d, d, bias=False)
        self.to_v = nn.Linear(d, d, bias=False)
        self.mix = nn.Linear(d, d, bias=False)

    def split(self, x):
        count, positions, d = x.shape
        return x.reshape(count, positions, self.heads, d // self.heads).transpose(1, 2)

    def forward(self, x, mask):
        q, k, v = self.split(self.to_q(x)), self.split(self.to_k(x)), self.split(self.to_v(x))
        out, self.weights = attention(q, k, v, mask)
        count, _, positions, _ = out.shape
        return self.mix(out.transpose(1, 2).reshape(count, positions, -1))


class Block(nn.Module):
    # One Transformer block: attention, add & norm, small MLP, add & norm (workshop c15, unchanged).

    def __init__(self, d, heads, dropout):
        super().__init__()
        self.attention = SelfAttention(d, heads)
        self.norm1 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d))
        self.norm2 = nn.LayerNorm(d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.norm1(x + self.dropout(self.attention(x, mask)))
        return self.norm2(x + self.dropout(self.mlp(x)))


def padding_mask(ids):
    # True where a word may be looked at: every real word; <pad> (id 0) is hidden. Shape (visits, 1, 1, positions)
    return (ids != 0)[:, None, None, :]


class VisitTransformer(nn.Module):
    # Click words in, one chance to come back and buy out.

    def __init__(self, vocab_size, block_size, d=32, heads=2, blocks=2, dropout=0.2):
        super().__init__()
        self.settings = {"vocab_size": vocab_size, "block_size": block_size, "d": d, "heads": heads,
                         "blocks": blocks, "dropout": dropout}
        self.word_embedding = nn.Embedding(vocab_size, d)          # what the click is   (workshop: letter_embedding)
        self.position_embedding = nn.Embedding(block_size, d)      # where it stands
        self.blocks = nn.ModuleList([Block(d, heads, dropout) for _ in range(blocks)])
        self.to_logit = nn.Linear(d, 1)                             # workshop: to_scores = Linear(d, vocab_size)

    def forward(self, ids):
        longest = int((ids != 0).sum(dim=1).max())                 # skip columns that are padding for every visit
        ids = ids[:, :longest]
        positions = torch.arange(longest, device=ids.device)
        x = self.word_embedding(ids) + self.position_embedding(positions)
        mask = padding_mask(ids)                                    # instead of causal_mask(positions)
        for block in self.blocks:
            x = block(x, mask)
        return self.to_logit(x[:, 0]).squeeze(-1)                   # the first position summarises the visit
