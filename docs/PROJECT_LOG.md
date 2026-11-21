# Nudge: project log

How the Nudge project was built: what we did, why, how, when, and with which tools. Read this before the presentation so you can answer the lecturer's questions.

- Course: IT102 Artificial Intelligence and Machine Learning, ISM University. Lecturer: Dr. Simonas Čepėnas.
- Report title: "Predicting Purchase Intent". Submission 4 December 2026, 23:00. Presentation 10 December 2026.
- Sources: `CLAUDE.md`, `docs/course-map.md`, `docs/tracking.md`, notebooks 01–04, `results/*.csv`, the code in `nudge/`, `scripts/`, `tests/`, and the git history. Every number below is copied from those files. Where two files disagree, we use the latest notebook output and say so.
- Course references: W3 = Week 3 slides, W6 = Week 6 (RNN), W8 = Week 8 (Transformers), "s16" = slide 16, WS = the Transformers workshop notebook, AK = review session answer key. Full map in `docs/course-map.md`.

---

## 1. Summary

About 70% of online carts are abandoned (Baymard: 70.22%, average of 50 studies). Shops answer with the same reminder, often with a 10% discount, to everyone. Hostinger Reach, Shopify and Klaviyo all work this way: none of them decides *who* gets the discount. So shops give margin away to people who would have come back anyway, and they teach shoppers to leave carts on purpose to get a coupon. **Nudge** is the missing decision layer. When a shopper leaves without buying, a sequence model reads their clicks in order and gives the chance they come back and buy within 7 days *on their own*. A decision layer turns that chance into: no message, reminder only, or reminder + 10%. We trained on 1,166,494 abandoned visits from 1 in 10 shoppers of a real multi-category shop (REES46, Oct–Nov 2019). On the same test shoppers, the Transformer reached AUC 0.682 (95% range 0.676–0.688), against 0.676 for logistic regression on word counts, 0.679 for the plain RNN and 0.558 for the industry rule. The gains are small but not luck: the paired test puts the Transformer above the RNN by +0.0034 (95% range +0.0018 to +0.0050). In money, the industry rule gives €50,327 of discount to test shoppers who came back anyway. The Transformer, sending discounts to 60% of cart abandoners, cuts that to €26,766 (range €23,718–30,382). The cost is that 3,051 cart abandoners who did not come back got no nudge, against 1,568 under the rule.

---

## 2. Timeline

Nearly all the work so far happened on **20 November 2026**. Times are local (Vilnius) from `git log`. Notebook run times come from the notebooks' saved execution stamps (UTC + 2 h).

| When | What happened | Evidence |
| --- | --- | --- |
| Before 20 Nov (date not recorded) | Talk with Lukas Jablonskas (Hostinger): most Hostinger sites are e-shops and small-business pages. He suggested an AI agent for e-shops, with WhatsApp for India and South-East Asia. Kristupas narrowed it to smarter abandoned-cart follow-ups. The "AI crawler detector" idea was dropped. | CLAUDE.md "Where the idea came from" |
| 20 Nov (time not recorded) | Idea, datasets and plan decided. Shared plan doc published. | CLAUDE.md "Current status" |
| 20 Nov 14:27 | Project skeleton: contract, decision layer, demo shell, notebook stubs (`155d143`) | git |
| 20 Nov 14:35 | GEMINI.md and AGENTS.md point to CLAUDE.md (`13f03eb`) | git |
| 20 Nov (time not recorded) | Private GitHub repo `kriksciunask/nudge` created and pushed. Local `.venv` (Python 3.13, torch 2.15 CPU, streamlit 1.67) installed; demo app tested. Kaggle API set up. | CLAUDE.md |
| 20 Nov (time not recorded) | First data look on Kaggle (`01_data`, Oct 2019, 1 in 150 shoppers): 281,621 clicks, 60,986 sessions. **No `remove_from_cart` in the data.** | CLAUDE.md |
| 20 Nov 15:00 | Data sample and richer click words (price band, `again`, time gap) (`a34776d`) | git |
| 20 Nov 15:07 | Script for the Data Explorer summary tables (`1f618d3`) | git |
| 20 Nov 15:24 | `away` click ending added (gap over 30 min) (`3ef0a8e`) | git |
| 20 Nov (time not recorded) | Decisions: 7-day come-back label; 1 in 10 shoppers, Oct + Nov; split by shopper 80/10/10; training only on the Mac | CLAUDE.md |
| 20 Nov (before 16:24) | **Independent data audit** by a separate reviewer; fixes built into `build_visits` | CLAUDE.md |
| 20 Nov 16:24 | Training data rebuilt after the audit (`b1c4f7d`). Full build takes 25 s on the Mac. | git, CLAUDE.md |
| 20 Nov 16:28 | Baselines: random, industry rule, logistic regression, money metric (`7b56940`) | git |
| 20 Nov 16:38–16:41 | `02_baselines` run in the lecturer's style | notebook stamps |
| 20 Nov 16:42 | Baselines rebuilt in the lecturer's style; `docs/course-map.md` written (`9ed8fc4`) | git |
| 20 Nov 17:03–17:32 | Demo shop "Mercato" built on branch `shop-mercato`: `app/app.py`, 90-product catalogue, Pexels photos. **Not committed yet.** | file times, `git status` |
| 20 Nov 17:12 | RNN and GRU in the Week 6 style (`83640de`) | git |
| 20 Nov 17:17–17:20 | `03_rnn` run again with the plain RNN only | notebook stamps |
| 20 Nov 17:20–17:33 | `04_transformer` run (training took 665 s) | notebook stamps |
| 20 Nov 17:38 | Transformer from the course workshop; GRU dropped (not taught) (`6eca85c`) | git |
| 20 Nov 17:39 | Result tables tracked in git (`c549e35`) | git |

Planned (CLAUDE.md "Timeline"): 24–28 Nov models and baselines; 27 Nov decide on model 2; 29 Nov–1 Dec demo and the five failures; 2–4 Dec slides, report, rehearsal.

---

## 3. Decisions log

All dates are 20 Nov 2026 unless the row says otherwise.

| Decision | Options considered | What we chose | Why | Date |
| --- | --- | --- | --- | --- |
| Dataset | REES46 multi-category store; REES46 cosmetics shop; Coveo SIGIR 2021; UCI Online Shoppers | **REES46 multi-category** for the main model | Real click logs with time, product, price, shopper and session. Cosmetics has `remove_from_cart`, so it is kept as a second-shop test. Coveo was skipped: sign-up wait and hashed products. UCI has no click order (17 summary features), so it is only a day-1 test. | 20 Nov |
| No `remove_from_cart` | Plain view/cart words; richer words | **Richer click words**: type + price band + `again` + time gap | Checked on all of October 2019: no remove events. 56% of buyers never add to cart first, so view/cart alone is too poor. Each word is type (view/cart) + price band (cheap < €100 / mid / expensive ≥ €250) + `again` (product already seen this visit) + gap (start / quick < 20 s / normal 20–70 s / long 70 s–30 min / away > 30 min). That gives 62 words (2 start words + 60 click words). | 20 Nov |
| `away` gap word | Treat long pauses as `long`; add a separate word | **Separate `away` word** (> 30 min) | Shop visit ids never reset; the longest in-visit gap was 333 h. Buy rate of visits with: quick 9.7%, normal 7.4%, long 6.4%, away 5.7%. | 20 Nov 15:24 |
| How much data | 1 in 150 shoppers, October only; larger | **1 in 10 shoppers** (`user_id % 10 == 0`), **October and November 2019** | Enough data to train; sampling by shopper keeps all of a shopper's visits together. Full files (~250 MB) built on Kaggle. | 20 Nov |
| Train / valid / test split | Random by row (as in the course, NB5 c11); by date; by shopper | **By shopper**, from an md5 hash of `user_id`: 80% train / 10% valid / 10% test. Extra check planned: train on October, test on November. | One shopper has many visits. A random row split would put the same shopper in train and test, and the model would recognise them (a leak). md5 gives the same split on every computer. The Oct→Nov check answers W6 s23 "never train on future data". | 20 Nov |
| Label | Same-visit label (did this visit end in a purchase?); 7-day come-back | **7-day come-back**: score only visits that end without a purchase; label 1 if the same shopper buys within 7 days after the decision moment | A real flow fires only when someone leaves without buying, so those are the visits to score. The business question is "would they come back and buy anyway?". The data has no messages, so a come-back is truly "without help". The same-visit label answered a different question and scored visits where no message would ever go out. The old `build_sessions` was kept for reference only. | 20 Nov |
| What a visit is | Shop's own session id; 30-min gap; 60-min gap | **60 minutes without a click**. Decision moment = last click + 60 min. | Session ids never reset (one lasted 16 days). 60 min matches Hostinger Reach, which checks abandoned carts every hour. A purchase inside that hour is the same visit and is not scored. Before this fix, 15% of "came back" cases bought within the first hour. | 20 Nov (audit) |
| `returning` start word | Any earlier visit in the data; a visit in the last 14 days | **A visit in the previous 14 days**; only visits from 15 Oct on are scored (1–14 Oct = history only) | With "any earlier visit", the returning share rose with the date (40% → 72%). With the fix it is steady at 62–73% per week. | 20 Nov (audit) |
| One-click visits | Drop them; keep them | **Keep them**; report their results separately (failure #1) | They are real abandoned visits that a shop would message. | 20 Nov |
| Money metric | Invented conversion rates; measurable numbers only | **€ wasted** = 10% of what the shopper actually bought when they came back, for shoppers who got a discount and came back anyway. **Left without a nudge** = cart abandoners we skipped who did not come back. At most **one discount per shopper per 7 days**, for every policy. | No invented numbers. Whether a discount would have converted the skipped shoppers is unknown; only an A/B test can say. | 20 Nov |
| Output and loss | 2 outputs + `CrossEntropyLoss` (course code, NB6 c22); 1 output + sigmoid + BCE (course slides, W3 s16–s18) | **1 output + sigmoid + binary cross-entropy** (`BCEWithLogitsLoss`) | Every model gives one number, the chance to come back and buy. Both are correct. Putting the sigmoid inside the loss is more stable (W3 s19: σ is flat at the extremes). | 20 Nov |
| Logistic regression tool | scikit-learn; PyTorch | **PyTorch** `nn.Linear` + sigmoid, trained with the same loop as the RNN and Transformer. scikit-learn kept only as a cross-check. | The course teaches logistic regression on slides only and never uses sklearn models. Cross-check: PyTorch AUC 0.676, scikit-learn 0.675. | 20 Nov 16:42 |
| GRU | Plain RNN; GRU; LSTM | **Plain RNN only**. GRU was tried, then removed. | The course names GRU/LSTM but never teaches their gate equations. The GRU did well (commit `83640de`: test AUC 0.682, PR-AUC 0.172), but we must be able to explain every model in course terms. | 20 Nov 17:38 |
| CPU or Apple GPU (MPS) | CPU; MPS | **CPU**, 4 threads | Measured in the notebooks: the CPU was about 10× faster than the Apple GPU for the RNN and about 2× faster for the Transformer, and results repeat exactly. | 20 Nov |
| Where training runs | Kaggle (free GPU); Kristupas's Mac | **Only Kristupas's Mac** (Apple M3, 16 GB). Kaggle only builds the data. | One fixed machine; models are small; data after building is 36 MB. | 20 Nov |
| Demo framework | Streamlit; other web tools | **Streamlit** | Same style as the workshop's `app.py`; runs from the GitHub repo in Python. | 20 Nov |
| Shop name | not recorded | **Mercato** | Reason not recorded. The app uses its own style, not a copy of any real brand (comment in `app/app.py`). | 20 Nov |
| Product photos | Openverse (no account); Pexels (free API key) | **Pexels** (Pexels License, credits saved in `app/shop/photo_credits.csv`) | Openverse was tried first (`scripts/fetch_product_photos.py`). Per the team, its photos were too poor; the reason is not written in any repo file. The Pexels script picks the most catalogue-like photo among the 5 most relevant results and never uses a photo twice. | 20 Nov 17:10–17:32 (file times) |

---

## 4. Data pipeline, step by step

| Step | Where it runs | What it does | Numbers |
| --- | --- | --- | --- |
| 0. First look | Kaggle, `01_data` (kernel `kriksciunas/nudge-01-data`) | October 2019, 1 in 150 shoppers | 281,621 clicks, 60,986 sessions; 6.5% of sessions buy; one-click sessions 37%; buy rate 10.4% if a product was viewed again vs 3.4% if not. Missing: `category_code` 33%, `brand` 14%. |
| 1. Sample | Kaggle, `01_data_build` (kernel `kriksciunas/nudge-01-data-build`) | Read `2019-Oct.csv` and `2019-Nov.csv` in pieces; keep shoppers with `user_id % 10 == 0` | `events_10pct.parquet`: 10,959,109 raw clicks (row count of the file, counted for this log) |
| 2. Clean | Mac, `clean_events` in `nudge/contract.py` | Remove exact duplicates and price ≤ 0; order clicks in the same second as view < cart < purchase | Count after cleaning: not recorded |
| 3. Cut into visits | Mac, `build_visits` | A visit ends after 60 min without a click | — |
| 4. Keep abandoned visits | `build_visits` | Drop visits with a purchase; keep only visits with full 14-day history and a full 7-day label window (decisions 15 Oct–23 Nov) | **1,166,494 abandoned visits** from 407,580 shoppers |
| 5. Label | `build_visits` | 1 if the same shopper buys within 7 days after last click + 60 min | 8.6% positive (100,863 visits). Cart abandoners: 89,417, of whom 21.1% come back anyway. Returning 11.5% vs new 3.3%. October 7.4% vs November 9.3%. |
| 6. Words | `tokens_from_clicks` (same function as the demo app) | Start word + one word per click | Words per visit: median 4, p99 45 |
| 7. Split | `split_of` (md5 of `user_id`) | 80 / 10 / 10 by shopper | Train 933,419 visits (8.7% positive); valid 115,544 (8.6%); test 117,531 (8.6%). 0 shoppers in two splits. |
| 8. Model input | `visits_to_ids` in `nudge/train.py` | Each visit → 64 word ids; padding 0 at the end; long visits keep the start word + last 63 clicks | Test matrix shape (117,531, 64); median 4 words, 90th percentile 15, max 64 |

The whole build (steps 2–7) is `scripts/build_training_data.py`. It took 25 s on the Mac. Output: `data/raw/visits_10pct.parquet` (36 MB, not in git). The team sample is `data/sample/visits_sample.parquet` (1 in 150 shoppers; 77,571 visits).

Earlier pilot numbers, before the audit (October sample, older rules): 44,770 abandoned visits, 10.6% come back and buy within 7 days; cart abandoners 26.8% (median 1.4 days); returning 14.1% vs new 4.4%. These are replaced by the full-data numbers above.

---

## 5. The independent data audit

A separate reviewer checked the data rules before any model was trained. Every fix is in `build_visits` (`nudge/contract.py`) and tested by `tests/test_contract.py`.

| # | Finding | Evidence | Fix |
| --- | --- | --- | --- |
| 1 | The shop's session id is not a visit | Session ids never reset; one lasted 16 days | A visit = clicks with no gap over 60 min. Decision moment = last click + 60 min (when Hostinger Reach emails). |
| 2 | "Came back" often meant "bought a few minutes later" | 15% of "came back" cases bought within the first hour | A purchase within 60 min is the same visit, which is not scored |
| 3 | Visits near the end of the data had no full 7-day window | Count of affected visits not recorded | Visits without a full 7-day window are dropped |
| 4 | "Returning" grew with the date | Returning share rose from 40% to 72% over the data | Returning = a visit in the previous 14 days; score only visits from 15 Oct. Now steady at 62–73% per week. |
| 5 | Cart value and money were wrong | Same product carted twice was counted twice; only about half of come-back shoppers buy something from the abandoned cart | Cart value counts each product once; `next_order_value` = what they actually bought on return; at most one discount per shopper per 7 days |
| 6 | Dirty rows | Exact duplicates, price ≤ 0, clicks in the same second in random order | Removed duplicates and price ≤ 0; ties ordered view < cart < purchase |
| 7 | Split depended on id digits used for sampling | Sampling uses `user_id % 10`; a split on digits would clash | md5 of `user_id`: 80/10/10, stable on every computer, 0 shoppers in two splits |
| 8 | Demo could send words the model never saw | Real shoppers open a product before carting it (99.6% in the data) | The app records a view before a cart click if there was none |

Still to do from the audit: bootstrap ranges on every number (done for quality and money; see section 7); Oct→Nov check, training only on visits that decide by 24 Oct; failure #3 becomes "shoppers who are not logged in are invisible".

---

## 6. Models

Every model is scored on the same 117,531 test visits. Every model picks its cut-off on validation (best F1), never on test (W3 s22). Shared code: `nudge/train.py` (loop), `nudge/evaluate.py` (scores). Seed 42. Times are the training cell durations saved in the notebooks.

| Model | What it is, in course terms | Formula and slide | Settings | Training time |
| --- | --- | --- | --- | --- |
| 0. Random | A coin flip: the floor | — | uniform random score, seed 42 | none |
| 1. Industry rule | "Left a cart → reminder + 10%" (Hostinger Reach, Shopify, Klaviyo) | score = 1 if the visit has a cart, else 0 (tiny noise breaks ties) | — | none |
| 2. Logistic regression | Perceptron with sigmoid on a **bag of words** (counts of each of the 62 words). Order is thrown away: Vignette E (AK p21). | z = Σ wᵢxᵢ + b, p = σ(z) = 1/(1+e^(−z)) (W3 s16–s17). Loss: L(w) = −Σᵢ [y⁽ⁱ⁾ log σ(z⁽ⁱ⁾) + (1−y⁽ⁱ⁾) log(1−σ(z⁽ⁱ⁾))] (W3 s18) | 63 parameters; Adam lr 0.01; batch 1,024; 8 epochs | 17 s |
| 3. Plain RNN (alternative specification) | Reads clicks one at a time and keeps a hidden state (memory). Embedding instead of one-hot (W3 s21, W8 s26). Final H_T → linear → sigmoid. | H_t = tanh(X_t W_xh + H_{t−1} W_hh + b_h) (W6 s28). Gradient clipping g ← min(1, θ/‖g‖) g, θ = 1 (W6 s31). Padding skipped with `pack_padded_sequence` (W8 s20). | 8,353 parameters; d = 32, hidden 64, dropout 0.2; Adam lr 0.001; batch 512; 6 epochs | 80 s |
| 4. Transformer (main model) | Encoder: every click attends to every other click. The workshop's `attention`, `SelfAttention` and `Block` are reused unchanged. Changes: a **padding mask** instead of the causal mask (W8 s20, s28); the first word (`new`/`returning`) is the summary, like `<cls>` (W8 s31); one logit + sigmoid. Learned position embedding X + P (W8 s25). | softmax(QKᵀ/√d)V (W8 s17–s20, WS c6). Block: attention → add & norm → MLP → add & norm (W8 s26–s27). | 20,929 parameters; d = 32, 2 heads, 2 blocks, dropout 0.2 (workshop settings); AdamW lr 0.001 (workshop used 0.003); batch 512; 6 epochs | 665 s (about 11 min) |

All neural models keep the weights of the epoch with the lowest validation loss (NB5 c22, WS c16). The padding mask was checked with a positive and a negative control: extra padding does not change the answer (True); an extra real click does change it (True).

---

## 7. Results

### 7.1 Quality on the test shoppers

Range in brackets = 95% range from resampling test **shoppers** 200 times.

| Model | AUC | PR-AUC | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| Random | 0.498 (0.492–0.504) | 0.085 (0.082–0.089) | 0.086 | 0.496 | 0.146 (0.140–0.152) |
| Industry rule | 0.558 (0.551–0.564) | 0.124 (0.118–0.130) | 0.209 | 0.184 | 0.196 |
| Logistic regression | 0.676 (0.670–0.682) | 0.157 (0.148–0.165) | 0.161 | 0.370 | 0.224 (0.215–0.233) |
| Plain RNN | 0.679 (0.673–0.685) | 0.163 (0.153–0.170) | 0.169 | 0.323 | 0.222 |
| **Transformer** | **0.682 (0.676–0.688)** | **0.168 (0.159–0.176)** | 0.166 | 0.359 | **0.227** |

- Ranges for precision and recall: not recorded. F1 ranges were recorded only in `02_baselines`.
- Random comes from `02_baselines`. The other rows come from `04_transformer` (the latest run).
- The industry rule scored 0.556 / 0.122 / F1 0.195 in `02_baselines` and 0.558 / 0.124 / F1 0.196 in `04_transformer`. The difference comes from the random tie-break noise. The table uses the latest run.
- Accuracy is not reported: 91.4% of visits do not come back, so "always no" is already 91.4% accurate.

### 7.2 Paired comparison (same shoppers resampled for both models)

| Difference in AUC | Value | 95% range | Clearly better? |
| --- | --- | --- | --- |
| Transformer − Logistic regression | 0.0062 | +0.0042 to +0.0087 | yes |
| Transformer − RNN | 0.0034 | +0.0018 to +0.0050 | yes |
| RNN − Logistic regression | 0.0028 | +0.0009 to +0.0051 | yes |

### 7.3 By visit length (AUC)

| Visit length (words) | Visits | Logistic regression | RNN | Transformer |
| --- | --- | --- | --- | --- |
| 1 click | 35,685 | 0.657 | 0.657 | 0.656 |
| 2–3 clicks | 31,725 | 0.658 | 0.657 | 0.662 |
| 4–7 clicks | 24,717 | 0.633 | 0.642 | 0.646 |
| 8–15 clicks | 15,726 | 0.634 | 0.628 | 0.638 |
| 16+ clicks | 9,678 | 0.637 | 0.636 | 0.646 |

Note: the groups are cut on the number of words including the start word, so "1 click" means 1–2 words.

### 7.4 Shuffle test (does the order of clicks matter?)

The clicks of each test visit are shuffled; the start word stays first.

| Model | AUC, real order | AUC, shuffled |
| --- | --- | --- |
| Logistic regression | 0.676 | 0.676 (cannot change: it only counts) |
| Plain RNN | 0.679 | 0.675 |
| Transformer | 0.682 | 0.680 |
| GRU (removed; from commit `83640de`) | 0.682 | 0.679 |

### 7.5 Money (test shoppers, cart abandoners only, 10% discount, max one per shopper per 7 days)

Each model ranks the cart abandoners and discounts the share it thinks are **least** likely to come back on their own.

| Policy | Share discounted | Discounts | € wasted (95% range) | Left without a nudge |
| --- | --- | --- | --- | --- |
| Industry rule (everyone) | 100% | 6,622 | 50,327 (44,510–56,529) | 1,568 |
| Logistic regression | 80% | 5,742 | 40,298 (35,384–45,417) | 2,234 |
| Random | 80% | 5,505 | 42,737 (37,431–48,432) | 2,502 |
| Logistic regression | 60% | 4,632 | 30,234 (26,407–34,453) | 3,128 |
| Random | 60% | 4,326 | 35,745 (30,970–40,977) | 3,498 |
| Plain RNN | 60% | 4,672 | 28,379 (24,640–31,770) | 3,052 |
| **Transformer** | 60% | 4,690 | **26,766 (23,718–30,382)** | 3,051 |
| Logistic regression | 40% | 3,369 | 19,756 (17,239–22,835) | 4,137 |
| Random | 40% | 3,047 | 26,242 (21,839–30,908) | 4,544 |

- The 80% and 40% rows and Random come from `02_baselines`. The 60% rows for LR, RNN and Transformer come from `04_transformer` (latest).
- Logistic regression at 60% was 4,619 / €30,202 / 3,138 in `02_baselines` and `03_rnn`, and 4,632 / €30,234 / 3,128 in `04_transformer`. The table uses the latest run.

### What the results mean

- **Order adds a little.** Reading clicks in order (RNN, Transformer) beats counting them (logistic regression). The paired test says the gain is real, but it is small: +0.0062 AUC for the Transformer over counts.
- **The Transformer helps most on longer visits** (4–7 clicks: 0.646 vs 0.633 for counts; 16+ clicks: 0.646 vs 0.637). On one-click visits all models are equal (about 0.657), because there is almost no order to read.
- **The real win is against today's rule.** The industry rule hardly separates shoppers (AUC 0.558). With 60% of the discounts, the Transformer wastes €26,766 instead of €50,327 on shoppers who came back anyway.
- **That saving has a cost.** 3,051 cart abandoners who did not come back got no nudge (1,568 under the rule). We do not know how many of them a discount would have won back.
- **Prediction is not causation.** Nudge predicts who comes back without help. Only an A/B test can prove that the discount causes extra sales: a random group of shoppers gets no message, and the rest are compared with them. We propose Hostinger runs this as the next step.
- **The data is old and short.** 2019 data, one shop, two months, including the Black Friday period of November (come-back rate 7.4% in October vs 9.3% in November). The Oct→Nov check is still to do.
- **If the complex model barely wins, that is an honest finding.** The GRU, which we removed because the course does not teach it, scored as well as the Transformer (AUC 0.682).

---

## 8. Tools and services

| Tool | What we used it for |
| --- | --- |
| Python 3.13 in a local `.venv` | All local code (checked: Python 3.13.7) |
| PyTorch 2.15 (CPU build) | Logistic regression, RNN, Transformer, training loop |
| pandas 3.1 | Tables of clicks and visits, parquet files |
| NumPy | Arrays, word counts, bootstrap |
| scikit-learn 1.9 | AUC and PR-AUC functions, confusion matrix; `LogisticRegression` only as a cross-check |
| matplotlib | Learning curves, heatmap, charts |
| Jupyter | The notebooks (kernel "nudge") |
| Streamlit 1.67 | Demo shop "Mercato" |
| Kaggle (Notebooks + API) | Reading the 2019 files and building the 1-in-10 sample; private notebooks `kriksciunas/nudge-01-data` and `kriksciunas/nudge-01-data-build` |
| Openverse, then Pexels API | Product photos (Pexels License; credits in `app/shop/photo_credits.csv`) |
| GitHub (private repo `kriksciunask/nudge`) | Code sharing; teammates invited with write access on 20 Nov |
| Claude Code with helper agents | Writing code with Kristupas, the independent data audit, the course map (`docs/course-map.md`), and this log |

**AI-use acknowledgement (ready to paste into the report):**

> We used an AI coding assistant (Claude Code, by Anthropic) throughout this project. It helped us write the Python code for the data pipeline, the models and the demo shop. It also ran an independent review of our data rules, mapped each part of the project to the course slides, and drafted the project log. Every decision about the problem, the data, the label, the models and the evaluation was made by the group and is recorded with its reasons in `CLAUDE.md` and `docs/PROJECT_LOG.md`. The main model was trained by us on our own data; no pre-trained model was used. Every group member reviewed the code and can explain how it works.

---

## 9. File map and how to rerun

Repo: `nudge-main` (the shop is on branch `shop-mercato` in the `hostinger_presentation_project` folder, not committed yet).

| What | Where |
| --- | --- |
| Project brief and all dated decisions | `CLAUDE.md` |
| Where each idea is taught | `docs/course-map.md` |
| What a shop must track | `docs/tracking.md` |
| Click words, data rules, `predict()` | `nudge/contract.py` |
| Slider → cut-offs → action | `nudge/decision.py` |
| Training loop, loaders | `nudge/train.py` |
| Scores, money, bootstrap | `nudge/evaluate.py` |
| Transformer (workshop code) | `nudge/transformer.py` |
| Build training data | `scripts/build_training_data.py` |
| Data checks | `tests/test_contract.py` |
| Notebooks | `notebooks/01_data`, `01_data_build` (Kaggle), `02_baselines`, `03_rnn`, `04_transformer`, `05_evaluation` (stub) |
| Results | `results/*_quality.csv`, `*_money.csv`, `*_by_length.csv`, `*.pt`, `results/scores/*.npy` |
| Data | `data/raw/` (not in git), `data/sample/` (team sample) |
| Demo shop | `app/app.py`, `app/shop/catalog.csv` (90 products), `app/shop/custom/products.csv` (10 classmate slots, see `app/shop/custom/README.md`), `app/shop/images/` |

Commands, run from the repo root:

```bash
# Check the data rules and the money metric (prints "all contract tests passed", "all evaluate tests passed")
.venv/bin/python tests/test_contract.py

# Rebuild the training data (needs data/raw/events_10pct.parquet from Kaggle; about 25 s)
.venv/bin/python scripts/build_training_data.py

# Re-run a notebook top to bottom (02, then 03, then 04: each reads the previous results)
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/02_baselines.ipynb

# Open the demo shop (browser at http://localhost:8501)
.venv/bin/python -m streamlit run app/app.py

# Fetch product photos (needs PEXELS_API_KEY in .env)
.venv/bin/python scripts/fetch_product_photos_pexels.py
```

Re-running a notebook trains a model, so run one at a time (the Transformer takes about 11 min).

---

## 10. Open items and next steps

| Item | Status |
| --- | --- |
| Tuning plan | Not recorded in CLAUDE.md. Ideas written down: optional `pos_weight` ≈ 10.6 for the rare positives (course-map section 2, not taught); the Oct→Nov check. |
| Oct→Nov robustness check | To do: train on visits that decide by 24 Oct, test on November |
| `05_evaluation` | Stub only: compare all models, profit at several cut-offs vs "discount everyone", the five failures |
| Five failures with real test examples | To do: (1) one-click visits; (2) returning customers (the model sees only this visit); (3) shoppers not logged in are invisible; (4) very long, indecisive visits; (5) expensive products |
| Plug the trained model into the app | To do: `predict()` in `contract.py` is still a marked placeholder; the app shows a warning banner |
| Commit the Mercato shop | To do: changes on `shop-mercato` are not committed |
| Two-screen demo | Planned after training: (1) shop-owner setup (discount size, margin, generosity, channel); (2) shopper view with a probability line that updates after every click, shaded zones, then the message on leaving |
| Data Explorer page | `scripts/make_explorer_data.py` still calls the removed `build_sessions` |
| Gustė's website comparison | Not recorded in any repo file |
| Model 2 (which channel and when) | Decide on 27 Nov |
| Backup demo video; test with people outside the group | Before 10 Dec |
| Team names in README; teammates accept GitHub invites | Open |

---

## 11. Explain it in 30 seconds

**Data.** "We used real click logs from a big online shop in 2019: 1 in 10 shoppers, October and November. A visit ends after an hour with no click, the same moment Hostinger Reach sends its email. We kept only visits that ended without buying, 1,166,494 of them. The label is 1 if the same shopper came back and bought within 7 days with no message; that is 8.6%. Each click became one word, such as 'view, expensive, seen again, long pause'. We split by shopper, 80/10/10, so no test shopper was ever seen in training. An independent audit fixed eight problems before we trained anything."

**Main model.** "The main model is a small Transformer encoder, built from the course workshop code. Every click looks at every other click with softmax(QKᵀ/√d)V. A position embedding keeps the order. We replaced the workshop's causal mask with a padding mask, because we score a finished visit. The first word, new or returning, works like the cls token and gives one probability. It has 21 thousand parameters and trained in about 11 minutes on a laptop CPU. Test AUC is 0.682."

**Comparison.** "We put four models against the Transformer on the same test shoppers: random, today's industry rule, logistic regression on word counts, and a plain RNN with the Week 6 formula. The rule gets AUC 0.558, counts 0.676, the RNN 0.679, the Transformer 0.682. The gains are small, but a paired bootstrap says they are not luck. Shuffling the clicks lowers the RNN and Transformer scores, so order carries some signal. The Transformer gains most on longer visits."

**Demo and GitHub.** "The demo is a Streamlit shop called Mercato with about 100 funny products. A visitor from the audience browses and adds to cart, and the panel shows the words Nudge sees. When they leave, Nudge shows the chance they come back alone and picks no message, a reminder, or a reminder with 10% off, with a WhatsApp preview. A slider sets how generous the shop is. Everything runs from our GitHub repo with one command."

**Story and slides.** "About 70% of carts are abandoned, and shops send everyone the same discount. That gives margin to people who would buy anyway and teaches shoppers to wait for coupons. Nudge decides who should get the discount. On our test shoppers, with 60% of the discounts, it wasted €26,766 instead of €50,327. The cost is that more people get no nudge. We predict; we do not prove that discounts cause sales. So the next step we propose to Hostinger is an A/B test."

---

## 12. Round 2: bigger sample, history words, tuning (night of 20–21 Nov)

**Why.** Round 1 showed small gains (AUC 0.676 → 0.682). Two things could add signal: more data and what the shopper did before this visit. We also had never tuned the settings; all models used the workshop defaults.

**Data.** Kaggle kernel `nudge-01-data-build` v3 built 1 in 4 shoppers (`user_id % 4 == 0`), Oct + Nov 2019: 27,497,039 clicks, 1.33 M shoppers. (v2 produced no file because the save cell had been removed by mistake when tidying the notebook; restored in commit 995ee41.) `scripts/build_training_data.py 25pct` on the Mac (104 s) gave **2,919,170 abandoned visits**, 8.7% come back and buy within 7 days (1-in-10 had 8.6%: consistent). Cart abandoners 224,482, 21.3% come back. Returning 11.5% vs new 3.3%. 0 shoppers in two splits.

**History words** (after the start word): `bought_recently`, `visits_1` / `visits_2_4` / `visits_5plus`, `last_today` / `last_1_3d` / `last_4_14d` — all from the previous 14 days only, so nothing from the future leaks in.

**Tuning (scripts/tune.py), judged only on validation shoppers.**
- Stage 1: every setting on a 25% sample of the training visits, up to 15 epochs, early stopping (patience 2), score = validation PR-AUC at the best epoch. Grid: LR 6, RNN 8, Transformer 14 settings → results/tuning/stage1.csv.
- Stage 2: the best 2 settings per model, on all training visits, seeds 42/43/44 → stage2.csv; models and val/test scores in results/tuned/. The test set is never used to choose anything.

**Stage 1 results (validation, 25% of train):**

| Model | Best setting | Val PR-AUC | Val AUC |
|---|---|---|---|
| Logistic regression | lr 0.003 (weight decay makes no difference) | 0.191 | 0.720 |
| RNN | d 16, hidden 128, lr 0.001 | 0.201 | 0.728 |
| Transformer | d 32, 3 blocks, dropout 0.1, lr 0.001 | 0.208 | 0.732 |

Round 1 test AUCs were 0.676 / 0.679 / 0.682; the jump to ~0.72–0.73 comes mostly from the history words (even LR, which has no order, gains 0.04). The order of the models stays the same: Transformer > RNN > LR, with small gaps. Stage 2 and the final test numbers: see notebooks/05_evaluation.ipynb (drafted the same night, reviewed for test-set leakage by a separate reviewer: none found).

**Mac care.** The chain waited for load < 8 and macOS memory pressure "free" ≥ 40% before each heavy step (the first version asked for 8 GB free+inactive, which macOS never shows because it fills spare memory with cache, so it waited for hours for nothing). One heavy job at a time, `caffeinate` so the Mac does not sleep.
