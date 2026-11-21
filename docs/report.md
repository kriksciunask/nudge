# Nudge: Predicting Purchase Intent

IT102 Artificial Intelligence and Machine Learning, ISM University. Lecturer: Dr. Simonas Čepėnas.
Group project, submitted 4 December 2026. Presented to Hostinger on 10 December 2026.

Course references follow `docs/course-map.md`: "Week 8 slide 20" means the lecturer's Week 8 slides, slide 20; "the workshop" is the course Transformers workshop notebook. Every number in this report comes from a file in `results/` or from `docs/PROJECT_LOG.md`.

---

## 1. The problem and who has it

About 70% of online carts are abandoned: the Baymard Institute puts the average at 70.22% across 50 studies. Many of those people were never going to buy that day. In Baymard's survey, 42% of US shoppers who abandoned a cart said they were "just browsing / not ready to buy".

Shops answer with an automatic follow-up. Hostinger's own tool, Hostinger Reach, has an abandoned-cart flow for its store builder: it checks for abandoned carts every hour and sends an email. Shopify and Klaviyo work the same way (Shopify can attach a discount code to every recovery email; Klaviyo sends the first email after 4 hours). None of these tools decides *who* should get the discount. Every abandoned cart is treated the same.

That costs the shop in three ways:

- **Wasted margin.** A discount sent to someone who would have come back and bought anyway is money given away. Example: on a €50 cart, 10% is €5. If that shopper had an 80% chance of buying anyway, the code wastes 0.8 × €5 = €4 on average.
- **Annoyed browsers.** People who were only looking get messages they did not want.
- **Trained coupon hunters.** If leaving a cart always brings a coupon, shoppers learn to leave carts on purpose. Klaviyo's own blog warns that repeat discounts train shoppers to wait.

Who has this problem: the small shops and e-shops that make up most Hostinger websites, and the shop owners using Hostinger Reach. Lukas Jablonskas (Hostinger) also pointed us to India and South-East Asia, where much e-commerce runs through WhatsApp: 93% of surveyed Indian SMEs sell on WhatsApp (Peak XV), and about 40% of Indonesian e-commerce transactions are informal, through WhatsApp, Instagram or Facebook (McKinsey, via Xendit).

## 2. The idea

Nudge is the decision layer that today's flows are missing. It has two parts.

**1. A prediction.** When a shopper leaves without buying, a sequence model reads their clicks in order and outputs one number: the chance they come back and buy within 7 days *on their own*.

**2. A decision.** That chance is turned into one of three actions (`nudge/decision.py`):

| Chance to come back alone | Action | Why |
| --- | --- | --- |
| Very low (below the low cut-off) | No message | Just browsing; a message would annoy |
| Middle | Reminder + 10% discount | Hesitant shopper; this is where a nudge can pay off |
| High (above the high cut-off) | Reminder only | Would buy anyway; a discount wastes margin |

The shop owner controls one setting: **"how generous are you with discounts"**, from stingy (0) to generous (1). It moves both cut-offs, which are set from the trained Transformer's scores on **validation** shoppers, never on test.

This slider is the precision/recall trade-off from the course. A wide discount band catches more of the shoppers who need a nudge (higher recall), but more discounts go to people who would have bought anyway (lower precision). A narrow band saves margin but misses more shoppers. The course treats this through the deployment rule "report class-specific errors and costs" (Week 3 slide 22); our money table in section 5 is that cost, in euros.

What we do not build: the sending itself (email or WhatsApp delivery, scheduling). Nudge sits on top of a tool like Hostinger Reach and tells it what to send to whom.

## 3. Data

### Source and sample

We used the REES46 "eCommerce behavior data from multi-category store" (Kaggle), a real click log from a large online shop selling mostly electronics. Each row is one event: time, type (view, cart, purchase), product, category, brand, price, shopper id and session id. The full data is about 15 GB, so a Kaggle notebook (`notebooks/01_data_build.ipynb`) kept every shopper with `user_id % 4 == 0` (1 in 4 shoppers) for October and November 2019: 27,497,039 clicks from 1.33 million shoppers. Sampling by shopper keeps all of a shopper's visits together. All later steps and all training ran on one laptop (Apple M3, 16 GB).

The data has no `remove_from_cart` events (checked on all of October 2019), so a click is either a view or a cart.

### From clicks to visits

The shop's own session ids never reset (one lasted 16 days), so we defined a visit ourselves:

- **A visit** = a shopper's clicks with no gap longer than 60 minutes.
- **Decision moment** = last click + 60 minutes. This is when Hostinger Reach, which checks hourly, would send its email.
- **We score only visits that end without a purchase.** That is when a real flow would fire. A purchase inside the 60 minutes belongs to the same visit, so that visit is not scored.

### The label: 7-day come-back

Label = 1 if the same shopper buys within 7 days after the decision moment, in a later visit. The data contains no marketing messages, so a come-back is truly "without help". This matches the business question: *would they come back and buy anyway?* Visits without a full 7-day window are dropped.

Result: **2,919,170 abandoned visits; 8.7% come back and buy within 7 days.** Among cart abandoners (224,482 visits), 21.3% come back on their own. Returning shoppers come back 11.5% of the time, new shoppers 3.3%.

### Click words

Each click becomes one word, built from four parts (`nudge/contract.py`):

| Part | Values |
| --- | --- |
| Type | `view`, `cart` |
| Price band | `cheap` (< €100), `mid`, `expensive` (≥ €250) |
| Seen before in this visit | `again` or nothing |
| Time since previous click | `start`, `quick` (< 20 s), `normal` (20–70 s), `long` (70 s–30 min), `away` (> 30 min) |

For example, `view_expensive_again_long` means "looked at an expensive product for the second time, after a long pause". That gives 60 click words.

Every visit starts with one shopper word, `new` or `returning` (a visit in the previous 14 days). In round 2 we added **history words** after it, all computed from the previous 14 days only, so nothing from the future leaks in: `bought_recently`; `visits_1`, `visits_2_4`, `visits_5plus`; `last_today`, `last_1_3d`, `last_4_14d`. The vocabulary is 69 words plus a padding word. A purchase is never part of the input.

A visit is therefore a list of words, like a sentence: `["returning", "visits_2_4", "last_1_3d", "view_expensive_start", "view_expensive_again_long", "cart_expensive_again_quick"]`. This is the course's token → index → embedding pipeline (Week 6 slide 24, Week 8 slide 26).

### Split by shopper

One shopper can have many visits. A random row split (as in the course notebooks) would put the same shopper in both training and test, and the model would partly recognise them. We split **by shopper** using an md5 hash of the shopper id: 80% train, 10% validation, 10% test. The hash gives the same split on every computer, and no shopper appears in two splits. The test set has **293,402 visits** from shoppers the models never saw.

### The independent data audit

Before any model was trained, a separate reviewer checked the data rules. The main findings and fixes:

| Finding | Fix |
| --- | --- |
| Session ids are not visits (one lasted 16 days) | A visit = no gap over 60 minutes |
| 15% of "came back" cases bought within the first hour | A purchase within 60 minutes is the same visit, not scored |
| "Returning" share grew with the date (40% → 72%) | Returning = a visit in the previous 14 days; score only visits from 15 October; now steady at 62–73% per week |
| Same product carted twice counted twice in the cart value | Each carted product counts once; money uses what the shopper actually bought on return |
| Duplicate rows, prices ≤ 0, same-second clicks in random order | Removed; ties ordered view < cart < purchase |
| Demo could send words the model never saw (99.6% of real shoppers view before carting) | The demo records a view before a cart click if there was none |

All fixes live in `build_visits` and are tested in `tests/test_contract.py`.

## 4. Models

We built a ladder of five models. Each is scored on the same test shoppers. If a more complex model only barely beats a simpler one, we report that as it is.

**0. Random.** Says "comes back" at the real rate. The floor.

**1. Industry rule.** "Left a cart → reminder + 10% to everyone." This is how Hostinger Reach, Shopify and Klaviyo behave. As a score: 1 if the visit has a cart, else 0.

**2. Logistic regression on word counts.** Counts how often each word appears in the visit and feeds the counts to a perceptron with a sigmoid (Week 3 slides 16–18):

  p = σ(z) = 1 / (1 + e^(−z)),  where z = Σ wᵢxᵢ + b

It throws away the order of the clicks, which is the lecturer's point in review Vignette E ("Read Between the Lines"): counts cannot tell "cart, then long pause, then leave" from the same clicks in another order. It is the fair test of whether order matters. Built in PyTorch as `nn.Linear` + sigmoid, with scikit-learn only as a cross-check (round 1: PyTorch AUC 0.676, scikit-learn 0.675).

**3. Plain RNN (the alternative specification).** The course requires an alternative to the Transformer. The RNN reads clicks one at a time and keeps a hidden state, in the lecturer's notation (Week 6 slide 28):

  H_t = tanh(X_t W_xh + H_{t−1} W_hh + b_h)

X_t is the learned embedding of click t. The final hidden state goes through a linear layer and a sigmoid. We use gradient clipping (Week 6 slide 31) and skip padding positions. We tried a GRU, which scored as well as the Transformer in round 1, and removed it: the course names GRU and LSTM but does not teach their gate equations, and we must be able to explain every model.

**4. Transformer encoder (the main model).** Every click looks at every other click through scaled dot-product attention (Week 8 slides 17–20):

  Attention(Q, K, V) = softmax(QKᵀ / √d) V

We reused the workshop's `attention`, `SelfAttention` and `Block` classes unchanged. Each block is attention → add & norm → small MLP → add & norm (Week 8 slides 26–27), and a learned position embedding is added to the word embedding (X + P, Week 8 slide 25) so the model knows the order. Three changes for our task:

- **Padding mask instead of causal mask.** The workshop generates names letter by letter, so each letter may only look back (the causal mask, Week 8 slide 28). We classify a finished visit, so every click may look at every other click; we only hide the padding (Week 8 slide 20). Checked with controls: extra padding does not change the answer; an extra real click does.
- **First-position readout.** The first word is always `new` or `returning`. Without a causal mask it can attend to every click, so its output works like the `<cls>` token (Week 8 slide 31): it becomes the summary of the visit.
- **One output.** The output layer gives one logit; sigmoid turns it into a chance. Loss is binary cross-entropy (Week 3 slide 18), computed on the logit for stability.

All neural models keep the weights from the epoch with the lowest validation loss, and cut long visits to the first word + the last 63 words.

### Tuning, on validation only

Every model got the same tuning effort (`scripts/tune.py`), and the test set was never used to choose anything (Week 3 slide 22: keep validation and test decisions separate).

- **Stage 1:** every setting trained on 25% of the training visits, one seed, up to 15 epochs with early stopping. Score = validation PR-AUC. Grid: 6 settings for logistic regression, 8 for the RNN, 14 for the Transformer (`results/tuning/stage1.csv`).
- **Stage 2:** the best 2 settings per model, retrained on all training visits with seeds 42, 43 and 44 (`results/tuning/stage2.csv`).
- **Final choice:** the setting with the highest mean validation PR-AUC over the three seeds. On test we average the three seeds' probabilities (a small ensemble that removes the luck of one seed). Each model's decision cut-off is the one with the best F1 on validation.

| Model | Final setting | Validation PR-AUC (3-seed mean) |
| --- | --- | --- |
| Logistic regression | lr 0.003, weight decay 0.0001 | 0.192 |
| Plain RNN | embedding d 16, hidden 128, dropout 0.2, lr 0.001 | 0.2015 |
| Transformer | d 32, 2 heads, 3 blocks, dropout 0.2, lr 0.001 | 0.2097 |

The seed spread is small (Transformer validation PR-AUC 0.2097 ± 0.0011), so the gaps between models are bigger than the noise from one seed.

## 5. Results

### Quality on the test shoppers

293,402 test visits; 8.8% come back and buy. Ranges are 95% ranges from resampling test **shoppers** 200 times (bootstrap).

| Model | AUC (95% range) | PR-AUC (95% range) | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| Random | 0.499 (0.496–0.503) | 0.087 (0.085–0.090) | 0.086 | 0.084 | 0.085 |
| Industry rule | 0.562 (0.559–0.565) | 0.112 (0.108–0.116) | 0.218 | 0.190 | 0.203 |
| Logistic regression | 0.716 (0.712–0.722) | 0.196 (0.188–0.206) | 0.193 | 0.455 | 0.271 |
| Plain RNN | 0.727 (0.722–0.733) | 0.204 (0.195–0.215) | 0.206 | 0.435 | 0.279 |
| **Transformer** | **0.729 (0.725–0.735)** | **0.213 (0.203–0.223)** | 0.203 | 0.451 | **0.280** |

How to read it:

- **AUC** = the chance that a random shopper who comes back gets a higher score than a random one who does not. 0.5 is a coin flip.
- **PR-AUC** focuses on the rare positives; a random model scores the base rate (about 0.087).
- **Accuracy is not reported.** About 91% of test visits do not come back, so "always no" is already about 91% accurate and useless.

### Is the more complex model really better?

Overlapping ranges do not settle it, so we use a **paired** bootstrap: each draw scores both models on the same resampled shoppers and keeps the difference.

| Difference | AUC (95% range) | PR-AUC (95% range) |
| --- | --- | --- |
| Transformer − RNN | +0.0024 (+0.0018 to +0.0029) | +0.0091 (+0.0071 to +0.0111) |
| Transformer − logistic regression | +0.0128 (+0.0115 to +0.0142) | +0.0172 (+0.0139 to +0.0205) |
| RNN − logistic regression | +0.0103 (+0.0091 to +0.0118) | +0.0080 (+0.0049 to +0.0114) |

Every range stays above zero, so each step up the ladder is real. But the steps are small. The honest summary: **reading clicks in order helps a little; the Transformer helps a little more than the RNN** (Week 8 slide 6 asks whether the complex model is worth it, and this is our answer). The big jump is from the industry rule (0.562) to any learned model (0.716 and up).

Where the gain came from: in round 1 (1 in 10 shoppers, no history words, workshop settings) the Transformer scored AUC 0.682. Most of the move to 0.729 came from the history words; logistic regression alone went from 0.676 to 0.716. Round 1 also showed that shuffling the clicks inside a visit lowers the RNN (0.679 → 0.675) and the Transformer (0.682 → 0.680), while logistic regression cannot change: order carries some signal, but not much.

### Who is in the top 10%?

A simpler way to say "it ranks well": among the 10% of test visits with the highest scores, how many come back?

| Model | Come-back rate in top 10% | Times the average |
| --- | --- | --- |
| Random | 8.5% | 0.96× |
| Industry rule | 18.5% | 2.1× |
| Logistic regression | 23.7% | 2.7× |
| Plain RNN | 24.0% | 2.7× |
| Transformer | **24.5%** (23.5–25.7%) | **2.8×** |

### Calibration

The decision layer uses fixed probability cut-offs, so the scores must roughly mean what they say. The calibration check by score decile was run on the second shop and is reported in section 7 (overall predicted 7.3% vs actual 7.9%). The result files do not include a decile table for the electronics test set.

### Money

We measure only what the data shows, with no invented conversion rates. Only **cart abandoners** count, because they are who today's flows discount. Each policy gives at most one discount per shopper per 7 days.

- **€ wasted** = 10% of what a shopper actually bought on return, for shoppers who got a discount and came back anyway.
- **Left without a nudge** = cart abandoners the policy skipped who did not come back. Whether a discount would have won them back is unknown.

Each model ranks the cart abandoners and discounts the share it thinks least likely to come back alone (the score that marks that share is taken from validation).

| Policy | Discounts sent | € wasted (95% range) | Left without a nudge |
| --- | --- | --- | --- |
| Industry rule (everyone) | 16,467 | 119,729 (110,392–128,016) | 4,080 |
| Random, 60% | 10,877 | 82,216 | 8,724 |
| Logistic regression, 60% | 11,932 | 72,317 | 7,410 |
| Plain RNN, 60% | 11,975 | 69,764 | 7,390 |
| **Transformer, 60%** | 11,987 | **68,785 (62,038–74,350)** | 7,315 |
| Transformer, 80% | 14,699 | 92,860 | 5,246 |
| Transformer, 40% | 8,503 | 42,293 | 10,146 |

At the 60% share, the Transformer wastes **€68,785 instead of €119,729: 43% less**. The cost is that 7,315 cart abandoners who did not come back got no nudge, against 4,080 under the rule. A shop can move along this line: more discounts, fewer people left out, more margin wasted. In the demo, the owner's slider moves along the same line. Its high cut-off is the 40th, 60th or 80th percentile of validation cart-abandoner scores at the stingy, middle and generous ends, so the middle setting is the 60% policy above. Its extra "just browsing" floor changes the middle-setting result by only €43 and 44 shoppers (€68,742 wasted, 7,359 left without a nudge; `results/money_curve.csv`).

### Robustness: train on October, test on November

The main split mixes both months. A real shop trains on the past and predicts the future, and November 2019 includes Black Friday (come-back rate 7.4% in October vs 9.3% in November). Week 6 slide 23 says "respect time: never train on future data". So we trained on visits deciding by 24 October and scored test shoppers' November visits. To separate "different month" from "less data", we also trained a reference model on a same-size random sample from both months.

| Transformer trained on | November AUC (95% range) |
| --- | --- |
| October only | 0.716 (0.709–0.721) |
| Both months, same size | 0.720 (0.713–0.724) |
| Both months, all visits | 0.722 (0.715–0.727) |

The month shift costs about 0.004 AUC and less data about 0.003. The model carries over to Black-Friday November.

## 6. Five failures

For each weak spot: the Transformer's results on that slice of the test set (`results/evaluation_failures.csv`). PR-AUC depends on the slice's come-back rate, so we also show PR-AUC ÷ rate ("times better than random").

| Slice | Visits | Come-back rate | AUC (95% range) | PR-AUC ÷ rate | Recall |
| --- | --- | --- | --- | --- | --- |
| All test visits | 293,402 | 8.8% | 0.729 (0.725–0.735) | 2.4 | 45% |
| 1. One click | 88,830 | 5.6% | 0.735 (0.726–0.743) | 2.5 | 25% |
| 2a. New shoppers | 100,458 | 3.4% | 0.674 (0.663–0.684) | 2.6 | 8% |
| 2b. Returning shoppers | 192,944 | 11.6% | 0.678 (0.672–0.684) | 2.0 | 51% |
| 3. Only one scored visit in our data | 52,001 | 1.6% | 0.747 (0.728–0.766) | 5.8 | 15% |
| 4. 30+ clicks | 7,547 | 19.8% | 0.675 (0.658–0.690) | 1.7 | 87% |
| 5. Any expensive product | 149,532 | 9.0% | 0.722 (0.716–0.728) | 2.4 | 46% |

1. **One-click visits (30% of visits).** The ranking holds (AUC 0.735), but recall is only 25%: with one click there is almost no order to read, so every sequence model falls back on the start word, the history words and that one click.
2. **New vs returning shoppers.** Within each group the model ranks worse (0.674 and 0.678) than overall, because much of the overall score comes from separating the two groups. New shoppers are the weakest slice: little to go on, and recall of 8%. The history words help returning shoppers, but the model sees nothing older than 14 days.
3. **Shoppers who are not logged in are invisible.** Shoppers with only one scored visit in our data come back just 1.6% of the time. Someone who browses on a phone and buys on a laptop without logging in looks like two people, so their purchase is missing and they count as "did not come back". No model can fix a label it cannot see. In production, logged-in customers (as in our demo) remove most of this.
4. **Very long, indecisive visits (30+ clicks).** Rare (2.6%) and mixed: serious comparison or aimless scrolling. AUC drops to 0.675, and the model only keeps the last 63 clicks.
5. **Expensive products.** Slightly below overall (0.722). Shoppers take longer over expensive items, so slow, repeated views mean something different there.

## 7. Second shop: cosmetics

**Question:** the Transformer was trained only on the electronics shop. Does it still rank shoppers in a different shop, with no retraining? (`notebooks/06_second_shop.ipynb`)

We built visits from the REES46 cosmetics shop with exactly the same rules: October–November 2019, 1 in 4 shoppers, 60-minute visits, 7-day label, history words. That gave 182,980 abandoned visits; 7.9% come back and buy. This shop does record `remove_from_cart` (19.3% of sampled events); we dropped those events, as the contract says. The prices are very different: 99% of cosmetics clicks are `cheap`, against about a third in electronics, so the model effectively sees one price word.

| Model (182,980 visits) | AUC (95% range) | PR-AUC (95% range) | Top-10% come-back rate |
| --- | --- | --- | --- |
| Random | 0.497 (0.492–0.502) | 0.078 | 7.6% |
| Industry rule | 0.716 (0.708–0.723) | 0.164 (0.158–0.169) | 21.3% |
| Logistic regression | 0.836 (0.832–0.839) | 0.254 (0.243–0.263) | 28.9% |
| **Transformer** | **0.841 (0.837–0.846)** | **0.288 (0.273–0.301)** | **31.2%** (4.0×) |

The ranking carries over. The paired gap over the rule is +0.125 AUC (+0.116 to +0.135). The higher AUC than on electronics does **not** mean the model is better at cosmetics: the industry rule also jumps (0.562 → 0.716), so this shop is simply easier to rank. The Transformer's gain over logistic regression stays small (+0.005), as on electronics.

**Calibration.** Mean predicted chance 7.3% vs actual 7.9%. The middle deciles are close (decile 7: predicted 7.0%, actual 7.6%; decile 8: 10.8% vs 11.5%). The extremes are stretched: the top decile is under-confident (25.6% predicted, 30.9% actual) and the bottom four are over-confident (about 1.5% predicted, 0.4–0.9% actual).

**Money with the electronics cut-offs.** The slider's cut-offs were set on electronics validation scores (middle setting: 5.5% and 21.5%). Used unchanged on cosmetics, the rule wastes €21,265 and Nudge €15,005 at the middle setting (−29%, 95% range €14,286–15,700), but Nudge leaves 15,325 non-returning cart abandoners without a nudge (rule: 8,951). The stingy end saves 53%, the generous end 11%. The saving is smaller than on electronics because cosmetics scores sit lower, so the same cut-offs behave differently: a shop owner would set the slider on their own shop's data.

**Honest framing.** This is a second shop from the same data company, with the same click format and the same months, not an independent platform. It tests the trained model, not the training recipe.

## 8. What the demo shows

The demo (`app/app.py`, Streamlit, run from the GitHub repo) is a made-up shop called Mercato. A member of the audience plays a logged-in customer.

- **The shop.** Browse products, open them, add to cart, remove from cart. Each click is turned into a click word with the same function that built the training data. Removing from the cart is a convenience in the shop only and is not sent to the model, which never saw that event.
- **Live chance meter.** After every click, the chance to come back alone is updated and drawn on a chart with shaded zones (no message / discount / reminder only), so the audience sees each click move the chance.
- **Leave the shop.** `predict()` scores the visit, the decision layer picks the action, and a WhatsApp-style message preview shows what the shopper would get.
- **The owner's setting.** The "how generous with discounts?" slider moves the cut-offs live, with a chart of discount money wasted against "discount everyone" (from `results/money_curve.csv`).
- **Look inside the model.** Two different questions, kept apart on purpose (`nudge/explain.py`):
  - *Where the model looked:* the attention weights softmax(QKᵀ/√d) of the last block, read at the first (summary) position, averaged over heads and seeds.
  - *Effect of this click:* remove one click, score the visit again, and show the change in percentage points (leave-one-out).

Attention is not an explanation. High attention on a click shows where the model looked, not that the click changed the answer (Jain & Wallace, 2019, "Attention is not Explanation"). The leave-one-out effect measures the output directly, so it is the honest check. We show both and say which is which.

## 9. Limits and the next step

- **Prediction is not causation.** Nudge predicts who comes back *without help*. It does not prove that a discount *causes* extra sales among the shoppers it targets. That needs an A/B test: a random group of shoppers gets no message, the rest get Nudge's decision, and the two are compared. **We propose that Hostinger runs this test as the next step**, with Hostinger Reach as the sending tool. This is also why any "extra sales" figure in the demo is a labelled assumption.
- **Logged-out shoppers are invisible** (failure 3). Cross-device purchases without a login are missed and labelled "did not come back".
- **Cut-offs per shop.** The cut-offs were set on electronics. The cosmetics test shows the ranking travels but the money does not: each shop should set its slider on its own data.
- **Old, short data.** One shop, two months of 2019, including Black Friday. The October → November check is reassuring, but a 2026 shop may behave differently.
- **Small gains from order.** The sequence models beat word counts by about 0.01 AUC. The case for Nudge rests on beating the industry rule, not on the Transformer beating the RNN.
- **Privacy.** WhatsApp marketing in the EU needs customer opt-in (GDPR), and production use needs the WhatsApp Business API. Only opted-in customers with a phone number can get WhatsApp messages. We mention this and do not build it.

## 10. AI use

We used an AI coding assistant (Claude Code, by Anthropic) throughout this project. It helped us write the Python code for the data pipeline, the models, the evaluation and the demo shop. It also ran an independent review of our data rules before training, checked the evaluation notebook for test-set leakage, mapped each part of the project to the course slides (`docs/course-map.md`), and drafted the project log and this report.

What the group did and checked:

- Made every decision about the problem, the data, the label, the models and the evaluation; each is recorded with its reasons in `CLAUDE.md` and `docs/PROJECT_LOG.md`.
- Checked the data rules with automated tests (`tests/`) and with positive and negative controls (for example, the padding-mask check in section 4).
- Trained every model ourselves on our own data. No pre-trained model was used; the Transformer reuses only the course workshop's code.
- Took every number in this report from the result files in `results/`.
- Reviewed the code; every group member can explain how it works.

## 11. References

- Baymard Institute, cart abandonment rate (70.22%, 50 studies; 42% "just browsing"): https://baymard.com/lists/cart-abandonment-rate
- Hostinger Reach abandoned-cart flow: https://www.hostinger.com/support/?p=8204
- Shopify, discounts for abandoned checkout recovery emails: https://help.shopify.com/en/manual/discounts/discounts-for-abandoned-checkout-recovery-emails
- Klaviyo, abandoned cart flow: https://help.klaviyo.com/hc/en-us/articles/115002779411
- Klaviyo, abandoned cart tips (repeat discounts train shoppers to wait): https://klaviyo.com/blog/abandoned-cart-tips
- Peak XV, Indian SMEs on WhatsApp: https://peakxv.com/?p=555
- Xendit, chat commerce in Indonesia (McKinsey figure): https://www.xendit.co/en/blog/sell-online-easily-with-chat-commerce/
- REES46, eCommerce behavior data from multi-category store: https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store
- REES46, eCommerce events history in cosmetics shop: https://www.kaggle.com/datasets/mkechinov/ecommerce-events-history-in-cosmetics-shop
- Jain, S. and Wallace, B. C. (2019). Attention is not Explanation. NAACL.
- Zhang, A., Lipton, Z. C., Li, M. and Smola, A. J. Dive into Deep Learning. https://d2l.ai
- Course materials: IT102 Week 3, Week 6 and Week 8 slides and the Transformers workshop notebook, Dr. Simonas Čepėnas, ISM University.
