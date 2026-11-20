# Course map: where the lecturer teaches each part of Nudge

Built 20 Nov 2026 from the files in `course/`. Every claim cites a file and a slide, page or notebook cell.

**Short names used below.** W2 = `AIML_Week#2_Full copy.pdf`, W3 = `AIML_Week#3.pdf`, W4 = `AIML_Week#4_training.pdf` (it is about CNNs, not training), W6 = `AI_ML_Week_6_RNN_2.pdf`, W8 = `AI_ML_Week_8_Transformers.pdf`, RS = `AI_ML_Review_Session.pdf`, AK = `AI_ML_Review_Session_AK.pdf` (answer key). "s16" = slide 16 (the same as PDF page 16). NB5 = `chapter5_mlp_student.ipynb`, NB6 = `chapter6_builders_guide_student.ipynb`, NB7 = `chapter7_cnn_student_1.ipynb`, WS = `Workshop_Transformers_students/1_name_generator.ipynb`. "c15" = notebook cell 15, counting from 0.

**Warning: `Chapter 2.pdf`, `Chapter 4.pdf` and `Chapter 5.pdf` are not Dive into Deep Learning.** They are chapters of a welfare-state economics book ("Rethinking welfare and the welfare state": Bismarck, Beveridge, public finance). So no textbook pages are cited here. Download the real d2l.ai chapters if you need them.

## 1. Where each part of Nudge is taught

| Our project step | Course concept (lecturer's term) | Where | Lecturer's notation, as written | Link to Nudge |
|---|---|---|---|---|
| A click becomes a word | Tokens, vocabulary, indices | W6 s24; W8 s11 | "token → index"; rare tokens become `<unk>` | Each click is a token. Our vocabulary has 62 words plus `<pad>` = 0. |
| Index → vector | One-hot vector | W3 s21; W6 s29 | label y=1 → [0,1,0]; "an index is a name, not a quantity" | Word id 17 is not "more" than id 3. That is why we never feed raw ids. |
| Index → learned vector | Embeddings | W8 s26, s34 ("What each token means (embeddings)"); WS c15 `nn.Embedding(vocab_size, d)` | tokens → embedding + position | Each click word gets a learned vector of length d. |
| A visit is a sequence | Sequence, time step, "sequence → one" | W6 s20, s21 | x₁, …, x_T; "varying length, order matters"; example "sentiment of a review" | One visit (any length) → one yes/no. The same shape as review sentiment. |
| Counts-only baseline | Logistic regression as a perceptron | W3 s16–s19; AK p19 (Vignette B) | z = Σ wᵢxᵢ, p = σ(z) = 1/(1+e^(−z)); ŷ = 1 if p ≥ 0.5 | Our LR on word counts (AUC 0.675). |
| Why counts are not enough | Vignette E "Read Between the Lines" | RS p18; AK p21 | "LR on word counts … throws away order" → RNN | The lecturer's own argument for our ladder: counts → RNN → Transformer. |
| Probability output | Sigmoid; softmax for q classes | W3 s16–s17, s20, s22; W8 s8 | σ(z); "Features → logits → softmax probabilities → decisions" | Our model ends in one logit → σ → chance to come back. |
| Loss | Negative log-likelihood = cross-entropy | W3 s18; W8 s8 | L(w) = −Σᵢ [y⁽ⁱ⁾ log σ(z⁽ⁱ⁾) + (1−y⁽ⁱ⁾) log(1−σ(z⁽ⁱ⁾))]; "−log ŷ_true class" | This is our binary cross-entropy loss. |
| Sigmoid gradient | Derivative of σ | W3 s19; RS Q17 | σ′(z) = σ(z)(1−σ(z)); "flat at the extremes, largest around 0.5" | Explains why we train on logits (stable). |
| Training | Forward pass, loss, backprop, gradient descent, epochs | W3 s26–s29; W4 s7, s31; AK p13 | "predict → measure the error → adjust the weights" | Same loop for RNN and Transformer. |
| Minibatches, optimizer, learning rate | DataLoader batches; Adam | NB5 c12–c13, c19; WS c17 | `Adam(lr=0.001)`; WS: `AdamW(lr=0.003)`, batch 64 | The term "minibatch SGD" is not found in the materials. |
| Train / valid / test | Three sets | NB5 c10; W2 s29; W3 s22 | "validation set: helps us choose the best epoch"; "Keep validation and test decisions separate" | We pick the cut-off and the best epoch on valid, then touch test once. |
| Overfitting | Overfitting, bias–variance | W2 s29–s30; AK p11–p12 | "complex enough … simple enough" | Watch train vs valid loss curves. |
| Weight decay | Simplicity penalty | W3 s13; RS Q10 | L_total = L_train + λ Σ wᵢ² | sklearn's `C` in our LR is this penalty (C = 1/λ). |
| Dropout | Dropout | W2 s32; NB5 c14; RS Q10 | `nn.Dropout(0.20)` | WS Transformer uses dropout 0.2. |
| Early stopping | "Keep the best validation weights" (the words "early stopping" are not found) | NB5 c22; WS c16 | "We keep the model from the lowest point of the orange line" | Keep the epoch with the best valid score. |
| MLP, activations | Hidden layers, ReLU, nonlinearity | W3 s25–s27; W6 s10–s12; W4 s27 | ReLU(z) = max(0, z); without it "layers collapse into one linear layer" | The feed-forward part of each Transformer block is a small MLP. |
| RNN | Hidden state | W6 s28; W8 s12; AK p15 | H_t = φ(X_t W_xh + H_{t−1} W_hh + b_h); O_t = H_t W_hq + b_q; φ = tanh | Our GRU baseline reads clicks one by one; the final H_T → sigmoid. |
| RNN training | BPTT, vanishing/exploding gradients, clipping | W6 s31; W8 s13; AK p15 | 0.5⁵⁰ ≈ 10⁻¹⁵; g ← min(1, θ/‖g‖) g | Early clicks of long visits fade. |
| GRU / LSTM | Gated RNNs (named only) | W6 s31; W2 s32; AK p15 ("Chapter 10") | No gate equations found | Gate formulas: not found in the materials. |
| Attention | Queries, keys, values; scaled dot product | W8 s17–s20; WS c5–c6 | softmax(QKᵀ/√d)V; worked example 0.67/0.24/0.09 | Each click looks at every other click. |
| Multi-head | h heads, concatenate, one linear layer | W8 s23; WS c15 `SelfAttention` | "Like channels in a CNN: many heads, many questions" | Two heads in the workshop model. |
| Order | Positional encoding (fixed waves or learned) | W8 s25 | p_{i,2j} = sin(i/10000^{2j/d}); X + P | We use the workshop's learned `position_embedding`. |
| Padding | Padding mask | W8 s20 (one line: "Padding is masked out") | — | Short visits are padded with `<pad>`; the mask hides them. |
| Causal mask | Masked self-attention, "no peeking" | W8 s26, s28; WS c8–c9 | lower-triangle table | Needed only to generate text. We classify a finished visit, so we drop it. |
| Transformer block | Self-attention → Add & norm → positionwise FFN → Add & norm | W8 s26–s27 | x + sublayer(x) | WS `Block` class does exactly this. |
| Whole-sequence label | `<cls>` token (vision Transformer) | W8 s31 | "its output becomes the label" | Our option: read the first position's output (see section 2). |
| Encoder for classification | Encoder-only (BERT): fine-tune to classify | W8 s22, s32 | "every token sees both sides" | Nudge is a small encoder-only classifier. |
| Metrics | Accuracy, precision, recall, F1 (named only); confusion matrix | W2 s29; RS Q9; NB5 c26–c27 | "non-differentiable … we optimize a surrogate" | Precision/recall formulas, ROC, AUC: not found. |
| Costs of errors | Deployment checklist | W3 s22 | "Report accuracy together with class-specific errors and costs" | Our money table. |
| Time order | "Respect time: never train on future data" | W6 s23; W8 s28 | — | See conflict note in section 4. |
| PyTorch parts | `nn.Module`, `__init__`/`forward`, `nn.Linear`, `nn.Sequential`, `DataLoader` | NB6 c15–c22, c13; NB5 c13 | "PyTorch creates the backward pass automatically" | Our models are `nn.Module` classes. |
| Save / load | Checkpoint dict | NB6 c74–c79; WS c23 | `torch.save({"config":…, "state_dict":…})` | `app/model.pt`. |

## 2. Reusing the workshop code

**What WS contains (cell 15 unless noted):** `attention(q, k, v, mask)` (c6), `causal_mask(size)` (c9), `pick_next(scores, temperature)` (c12), `SelfAttention(d, heads)` with `to_q`, `to_k`, `to_v`, `mix` and stored `self.weights`, `Block(d, heads, dropout)`, `NameGPT(vocab_size, block_size, d=32, heads=2, blocks=2, dropout=0.2)` with `letter_embedding`, `position_embedding`, `blocks`, `to_scores`, `generate(...)`, `new_model()`. Training: `loss_on` and `train(model, steps=2000, batch_size=64, learning_rate=0.003)` with `AdamW` (c17). Heatmap of `model.blocks[-1].attention.weights[0]` (c21). Save to `app/model.pt` (c23). The app (`app/app.py`) copies the same classes (lines 32–100), loads the model with `@st.cache_resource` (line 105) and draws an Altair attention table (`attention_table`, line 146).

**Settings it uses:** `SETTINGS = {"d": 32, "heads": 2, "blocks": 2}`, dropout 0.2, AdamW lr 0.003, 2,000 steps of 64 random rows, validation every 100 steps, keep the best-validation weights, `SEED = 42`, `torch.set_num_threads(1)`. The MLP inside `Block` is `Linear(d, 2d) → ReLU → Linear(2d, d)`. The block is "post-norm": `norm1(x + dropout(attention))`. The split is a random 90/10 by word (c4).

**Keep unchanged:** `attention`, `SelfAttention`, `Block`, the embedding + position embedding line, the training loop shape, the save cell, the heatmap cell.

**Change for Nudge:**

1. **Vocabulary.** `chars`/`char_to_id` → `VOCAB`/`TOKEN_TO_ID` from `nudge/contract.py` (63 entries; id 0 = `<pad>`). In WS, id 0 is the start/end symbol "."; for us, 0 means padding only.
2. **Data rows.** WS `encode` makes targets Y shifted by one letter. We need X = word ids padded to `BLOCK_SIZE = 64` (last 64 words) and one label y ∈ {0, 1} per row.
3. **Mask.** Replace `causal_mask(positions)` with a padding mask: `mask = (ids != 0)[:, None, None, :]` (True = may look, the same meaning as in WS). `attention` works unchanged because `masked_fill` broadcasts. Every visit has at least its start word, so no row is fully hidden.
4. **One answer per visit (pooling).** Two options. (a) Masked mean: average the output vectors of the real words. (b) First position, like `<cls>` in W8 s31: our first word is always "new"/"returning", and without a causal mask it can look at every click, so its output can serve as the summary. Option (a) is simpler to defend; (b) gives a clean heatmap row ("which clicks did the summary look at?").
5. **Output and loss.** `to_scores = nn.Linear(d, vocab_size)` → `nn.Linear(d, 1)`. `F.cross_entropy` → `F.binary_cross_entropy_with_logits(logit, y)`, which is the W3 s18 formula with σ inside. For 8.6% positives, an optional `pos_weight` (≈ 10.6 = negatives/positives) up-weights buyers; this is not taught (section 4).
6. **Delete** `pick_next`, `generate`, temperature: we never generate.
7. **Training size.** 933k training visits vs ~3,000 names: switch from 2,000 random-row steps to epochs over a `DataLoader` (NB5 style), and check valid on a fixed sample.
8. **Heatmap.** Same `imshow` code, but the full square is filled (no empty triangle). In the app's `attention_table`, remove the `if j <= i` filter.

## 3. Notation cheat-sheet (use these in the report)

| Symbol | Meaning | Source |
|---|---|---|
| x₁, …, x_T; t; T | tokens of one sequence, time step, length | W6 s20 |
| X_t | input at step t (row vector, one-hot or embedding) | W6 s28 |
| H_t | hidden state = memory (not "hidden layer") | W6 s28 |
| W_xh, W_hh, b_h | input→hidden, hidden→hidden weights, bias | W6 s28 |
| W_hq, b_q, O_t | hidden→output weights, bias, output | W6 s28 |
| φ | activation, e.g. tanh | W6 s28 |
| z, w, b, σ(z), p, ŷ | net input, weights, bias, sigmoid, probability, prediction | W3 s15–s16 |
| h(x) = σ(wᵀx + b) | "hypothesis" | W3 s17 |
| L(w), λ | loss, weight-decay strength | W3 s13, s18 |
| q, kᵢ, vᵢ, α(q, kᵢ) | query, key, value, attention weight | W8 s17 |
| Q, K, V, d | stacked queries/keys/values; vector length | W8 s20 |
| h | number of heads | W8 s23 |
| P, p_{i,j} | positional encoding added as X + P | W8 s25 |
| n | number of tokens (n² cost) | W8 s24 |
| \|V\| | vocabulary size | W6 s26 |
| θ | clipping threshold | W6 s31 |

The lecturer writes **d**, not "d_model" ("d_model" is not found). He writes the RNN with row vectors: X_t W_xh, not W·x. Our CLAUDE.md formula "h_new = tanh(W·x + U·h_old + b)" should become H_t = tanh(X_t W_xh + H_{t−1} W_hh + b_h).

## 4. Gaps (used by Nudge, not taught)

**Precision, recall, F1.** Named only (W2 s29, RS Q9), never defined. Precision = of the shoppers we flag, the share who really come back (TP/(TP+FP)). Recall = of the shoppers who come back, the share we flag (TP/(TP+FN)). F1 = their harmonic mean, 2PR/(P+R). With 8.6% positives, accuracy is misleading: "nobody comes back" is 91.4% accurate and useless.

**ROC-AUC and PR-AUC.** Not found. AUC = the chance a random come-back shopper gets a higher score than a random non-buyer (0.5 = coin flip). It does not depend on a cut-off. PR-AUC (average precision) focuses on the rare positives; a random model scores the positive rate (0.086).

**Class imbalance.** Not found; the notebooks only use `stratify=y` in `train_test_split` (NB5 c11). Our answer: report PR-AUC and F1, choose the cut-off on valid, optionally `pos_weight`.

**Cut-off chosen on validation.** The course uses p ≥ 0.5 (W3 s16). With rare positives almost no one passes 0.5, so we pick the cut-off with the best F1 on valid. This follows W3 s22 "Keep validation and test decisions separate".

**Split by shopper with a hash.** Not found; the course splits rows randomly (NB5 c11; WS c4). One shopper has many visits. A random split would put the same shopper in train and test, and the model would recognise them (leak). md5 of `user_id` gives a fixed 80/10/10 split on every computer.

**Time order (possible conflict).** W6 s23 says "Respect time: never train on future data". Our main split mixes October and November shoppers. Defence: no test shopper is ever seen in training, and we add the Oct→Nov check (train on October, test on November).

**Bootstrap ranges.** Not found. We resample test shoppers 200 times and recompute each number; the middle 95% is the range. It shows whether a gap between two models is bigger than chance.

**AdamW, logits loss, gate equations.** AdamW and Adam appear in code only, unexplained (WS c17; NB5 c19). GRU/LSTM gates: not found. Masked mean pooling: not found.

**Prediction vs causation.** Not found as such. Nudge predicts who comes back; only an A/B test can prove a discount causes extra sales.

## 5. Likely lecturer questions

1. **Why not logistic regression on counts?** That is Vignette E (AK p21): counts throw away order. Our LR gets AUC 0.675; the RNN and Transformer show how much order adds.
2. **Why a Transformer, not only an RNN?** W8 s14: an RNN carries everything in one H_t and the signal fades over long chains (W6 s31). Self-attention links any two clicks in one step, O(1) path (W8 s24). We compare the two on short vs long visits.
3. **Write the attention formula and explain √d.** softmax(QKᵀ/√d)V (W8 s20). Dividing by √d keeps scores on one scale; softmax makes weights positive and sum to 1. Each click becomes a blend of the clicks that matter to it.
4. **Why no causal mask?** It stops a decoder peeking at future tokens while generating (W8 s28). We score a finished visit, so every click may see every other, like an encoder (W8 s22). We still mask padding (W8 s20).
5. **How does the model know the order?** Self-attention alone cannot (W8 s25: "dog bites man"). We add a learned position embedding, X + P (WS c15).
6. **Hidden state vs hidden layer?** H_t is carried across time steps; a hidden layer sits between input and output (W6 s28; AK p15). The same weights are reused each step, so a 2-click and a 45-click visit use the same parameters.
7. **Which loss and why?** Cross-entropy / negative log-likelihood (W3 s18): it rewards a high probability for the true label. Accuracy and F1 are not differentiable, so we train on this surrogate (W2 s29).
8. **How do you know you are not overfitting?** Separate valid set, learning curves, dropout 0.2, keep the best-valid weights (NB5 c22–c24; WS c16), weight decay in LR (W3 s13). Test is used once.
9. **Why not accuracy?** 91.4% of visits do not come back; "always no" gets 91.4%. W3 s22 asks for class-specific errors and costs: we report precision, recall, F1, AUC and euros.
10. **Is the complex model worth it?** That is the W8 s6 baseline question. We report every model on the same test shoppers with 95% ranges; if the Transformer barely wins, we say so.
11. **Five failures?** W8 s6: one-click visits, returning customers (the model sees only this visit), shoppers not logged in, long indecisive visits, expensive products. Each needs a real test example and a reason.
12. **What is learned and what did you choose?** Learned: embeddings, Q/K/V projections, MLP weights (W8 s34). Chosen: d=32, 2 heads, 2 blocks, dropout, the click vocabulary (AK p13 style answer).

## 6. How the lecturer writes code

**Notebook skeleton (NB5, NB6, NB7).** `# Chapter N Problem: <topic>` → "This notebook follows the main ideas in Chapter N of *Dive into Deep Learning*" + numbered list → `## What you will do` (**Guided problem** / **Independent problem**) → `## 0. One-time setup` (a commented `# %pip install numpy matplotlib scikit-learn torch`, NB5 c2) → `## 1. Import the packages` with one bullet per package (NB5 c3) → numbered `## 2.`, `## 3.` … sections, each a short markdown cell then one code cell → `## N. Connect the code to Chapter N`: a table "idea | where it appears here" (NB5 c30, NB6 c82) → `# Independent problem` with `# TODO 1: …` cells (NB5 c34–c40) → `## Exercise checklist` → `## Reflection questions` → `## Submission` (NB5 c41). NB7 uses "Part A (guided)" / "Part B … Mission 1–8". The workshop uses `TODO` with `...` to fill in, then a `CHECK` cell that prints `CORRECT.` or `NOT YET: …` (WS c6–c7).

**Imports (NB5 c4):**
```python
from sklearn.datasets import load_digits, make_moons
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
SEED = 42
```
Plus `numpy as np`, `matplotlib.pyplot as plt`, `torch`, `copy`, `random`; `torch.nn.functional as F` in NB7 c4 and WS c2. **Not used:** `d2l` (not imported anywhere), pandas (only in the workshop app), custom `Dataset` classes, tqdm, any sklearn model (no `LogisticRegression`), BCE loss, `weight_decay=`, GRU/LSTM code.

**Logistic regression is taught only on slides** (W3 s16–s19 math; W4 s15 worked example σ(1.4) ≈ 0.80; AK p19). No code for it is found, neither sklearn nor PyTorch. To match the course style, rebuild our LR as a PyTorch module, `nn.Linear(63, 1)` on word counts + sigmoid, trained with the same `train_model` loop. sklearn's `LogisticRegression` is still fine as a cross-check.

**Binary classes in code = 2 logits + `CrossEntropyLoss`**, not sigmoid + BCE (NB6 c22 `self.output = nn.Linear(16, 2)`; `chapter5_mlp_student_answer_key.pdf` p16, two-moons: `nn.Linear(16, 2)`). The slides teach sigmoid + BCE (W3 s16–s18). Both are correct; pick one and say why.

**Data loading (NB5 c13):**
```python
def make_loader(features, targets, batch_size=64, shuffle=False):
    dataset = TensorDataset(feature_tensor, target_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
```
Splits: `train_test_split(..., test_size=0.15, random_state=SEED, stratify=y)` twice → `X_train, X_val, X_test` (NB5 c11).

**Models:** a class `DigitMLP(nn.Module)` with `self.network = nn.Sequential(...)` (NB5 c15); reusable blocks (`HiddenBlock`, NB6 c20); settings in a `config` dict passed as `**config` (NB6 c67); Xavier init via `model.apply(initialize_weights)` (NB5 c17); parameter count printed with `sum(p.numel() for p in model.parameters())`.

**Training (NB5 c19, c21):**
```python
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
...
optimizer.zero_grad(); logits = model(features)
loss = loss_function(logits, targets); loss.backward(); optimizer.step()
```
Functions `evaluate(model, loader, loss_function)` → (loss, accuracy) and `train_model(model, train_loader, val_loader, loss_function, optimizer, epochs=25)` returning a `history` dict with keys `train_loss`, `train_accuracy`, `val_loss`, `val_accuracy`. Best weights kept with `copy.deepcopy(model.state_dict())`. Printing: `f"Epoch {epoch + 1:2d}/{epochs} | train loss {train_loss:.4f} | ..."` every 5 epochs. Device: `try_gpu()` returns cuda, mps or cpu (NB6 c62).

**Results:** matplotlib two-panel learning curves, "Training" vs "Validation", `ylabel="Cross-entropy loss"` (NB5 c25; NB6 c71 `plot_history`); test evaluated once, printed as `f"Test accuracy: {test_accuracy:.3%}"`; `ConfusionMatrixDisplay.from_predictions(..., cmap="Blues", colorbar=False)` (NB5 c27); a grid of right/wrong examples in green/red (NB5 c29). Save: `torch.save({"config":…, "state_dict":…}, path)` (NB6 c77).

**Naming:** full words (`loss_function`, `features`, `targets`, `predictions`, `number_of_parameters`), `X_`/`y_` for data, upper case for constants (`SEED`, `VOCAB_SIZE`, `BLOCK_SIZE`), docstrings in plain sentences, comments that name the course idea (`# 4. how to change`).
