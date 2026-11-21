"""Look inside the Transformer: where does the first (summary) position look, and what really changes the answer?

Two different questions, kept apart on purpose (Jain & Wallace 2019, "Attention is not Explanation"):
1. WHERE THE MODEL LOOKED: the attention weights softmax(QK^T / sqrt(d)) of the last block (Week 8 slide 20),
   read at the first position, the <cls>-style summary (Week 8 slide 31). Averaged over 2 heads and 3 seeds.
2. WHAT CHANGES THE ANSWER: leave one click out, score the visit again; the effect of the click = chance with it minus chance without it.
   This is the honest check: it measures the output, not the inside.

All functions take the same list of words as predict() (nudge/contract.py) and cut long visits the same way
(first word + last BLOCK_SIZE - 1 words). They return one value per KEPT word, see `kept()`.
"""

import numpy as np
import torch

from nudge.contract import BLOCK_SIZE, _load_models, encode, predict


def kept(words: list[str]) -> list[str]:
    """The words the model really reads: all of them, or for a long visit the first word + the last 63."""
    return list(words) if len(words) <= BLOCK_SIZE else [words[0]] + list(words[-(BLOCK_SIZE - 1):])


def _last_block_attention(words: list[str]) -> np.ndarray:
    """(positions, positions): last block, averaged over heads and over the 3 seeds. Row = who looks, column = looked at."""
    x = torch.tensor([encode(kept(words))])
    grids = []
    with torch.no_grad():
        for model in _load_models():
            model(x)
            weights = model.blocks[-1].attention.weights      # (1 visit, heads, positions, positions), saved by SelfAttention
            grids.append(weights[0].mean(dim=0).numpy())      # average over heads
    return np.mean(grids, axis=0)                              # average over seeds; every row still sums to 1


def attention_matrix(words: list[str]) -> np.ndarray:
    """The full word x word attention grid of the last block (rows sum to 1)."""
    return _last_block_attention(words)


def attention_to_summary(words: list[str]) -> list[float]:
    """How much the first (summary) position attends to each word. One weight per word, sums to 1."""
    return [float(w) for w in _last_block_attention(words)[0]]


def effect_of_each(words: list[str]) -> list[float | None]:
    """Effect of each CLICK on the chance, in percentage points: chance with it minus chance without it (leave-one-out).
    Positive = that click raised the chance, like the live meter's up arrow. None for words that are not clicks:
    the start word and the history words cannot be removed."""
    words = kept(words)
    base = predict(words)
    out: list[float | None] = []
    for i, word in enumerate(words):
        if i == 0 or not word.startswith(("view_", "cart_")):
            out.append(None)
        else:
            out.append(100 * (base - predict(words[:i] + words[i + 1:])))
    return out
