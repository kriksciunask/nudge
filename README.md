# Nudge – Predicting Purchase Intent

IT102 Artificial Intelligence and Machine Learning, ISM University (group project). Lecturer: Dr. Simonas Čepėnas.

About 70% of online carts are abandoned (Baymard Institute), and shop tools such as Hostinger Reach, Shopify and Klaviyo
send the same reminder, often with a 10% discount, to every abandoned cart. That gives margin away to people who would
have bought anyway, annoys people who were only browsing, and teaches shoppers to leave carts on purpose to get a coupon.
**Nudge decides who should get the discount.** When a shopper leaves, a Transformer encoder (with a plain RNN as the
alternative specification) reads their clicks in order and gives the chance they come back and buy within 7 days on
their own. A decision layer turns that chance into **no message**, **reminder + 10%**, or **reminder only**. The shop
owner sets one "how generous are you with discounts" slider. Full write-up: [`docs/report.md`](docs/report.md).

## Run the demo

You need git and Python 3.12 or 3.13 (not 3.15: PyTorch may not support it yet).

```bash
git clone https://github.com/kriksciunask/nudge.git
cd nudge
python3.13 -m venv .venv          # or python3.12
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt   # the demo only; notebooks and training also need: pip install -r requirements-dev.txt
python -m streamlit run app/app.py
```

The browser opens at http://localhost:8501. Next time: `cd nudge`, `source .venv/bin/activate`, `git pull`, then the last line.

## What you'll see in the demo

- **The shop.** A made-up shop, Mercato. You are a logged-in customer: browse, open products, add to cart, remove from cart.
- **Live chance meter.** After every click, the chance that you come back and buy on your own updates on a chart with three zones: no message, discount, reminder only.
- **Leave the shop.** Nudge scores your visit, picks the action, and shows a WhatsApp-style preview of the message you would get.
- **Look inside the model.** Where the Transformer looked (attention weights of its summary position) next to the **effect of this click** (remove one click, score again, see how much the chance moves). Attention shows where the model looked; the effect shows what changed the answer.
- **The owner's setting.** The "how generous with discounts?" slider moves the cut-offs, with a chart of discount money wasted compared with "discount everyone".

## Results

Test set: 293,402 visits from shoppers the models never saw in training (REES46 electronics shop, Oct–Nov 2019).
8.8% of these visits come back and buy within 7 days. Ranges are 95% bootstrap ranges by shopper.

| Model | AUC | PR-AUC |
| --- | --- | --- |
| Random | 0.499 | 0.087 |
| Industry rule ("discount every cart") | 0.562 | 0.112 |
| Logistic regression on word counts | 0.716 | 0.196 |
| Plain RNN | 0.727 | 0.204 |
| **Transformer** | **0.729** (0.725–0.735) | **0.213** (0.203–0.223) |

- **Money:** discounting 60% of cart abandoners, the Transformer gives €68,785 to shoppers who came back anyway, against €119,729 under the industry rule (**−43%**). The cost: 7,315 cart abandoners who did not come back got no nudge (rule: 4,080).
- **Second shop:** with no retraining, on a REES46 cosmetics shop (182,980 visits), the Transformer reaches AUC 0.841 against 0.716 for the rule. That shop is easier for every model, so the fair claim is "it still clearly beats the rule".
- **Honest limit:** Nudge predicts who comes back without help. Proving that a discount *causes* extra sales needs an A/B test, which we propose Hostinger runs next.

## Repo map

| Path | What's in it |
| --- | --- |
| `nudge/contract.py` | Click words, data rules (`build_visits`), `predict(session) -> float` |
| `nudge/decision.py` | Slider → cut-offs → action |
| `nudge/models.py`, `nudge/transformer.py` | Logistic regression, plain RNN, Transformer (workshop code) |
| `nudge/train.py`, `nudge/evaluate.py` | Training loop; scores, money, bootstrap |
| `nudge/explain.py` | "Look inside": attention and leave-one-out effect |
| `notebooks/01_data`, `01_data_build` | First look at the data; building the sample (run on Kaggle) |
| `notebooks/02_baselines` | Random, industry rule, logistic regression |
| `notebooks/03_rnn`, `04_transformer` | Plain RNN; Transformer |
| `notebooks/05_evaluation` | Final comparison, money, October → November check, five failures |
| `notebooks/06_second_shop` | Cosmetics shop test, no retraining |
| `scripts/build_training_data.py` | Kaggle clicks → visits with labels |
| `scripts/build_cosmetics_visits.py` | Same for the cosmetics shop |
| `scripts/tune.py` | Tuning, judged on validation shoppers only |
| `scripts/make_money_curve.py` | Money for every slider setting (`results/money_curve.csv`) |
| `scripts/fetch_product_photos*.py`, `make_explorer_data.py` | Demo photos; Data Explorer tables |
| `app/` | Streamlit demo shop and the trained `model.pt` |
| `results/` | Result tables (`evaluation_*.csv`, `second_shop_*.csv`, `tuning/`) |
| `tests/` | Checks for the data rules, evaluation, explain and app |
| `data/` | Where the data comes from (raw data is not committed) |
| `docs/` | `report.md`, `PROJECT_LOG.md` (full record), `course-map.md` (link to the course slides), `tracking.md` |

Notebooks 02–06 and all training run on a laptop; Kaggle only builds the data sample.

## Team

| Role | Name |
| --- | --- |
| Data | TBD |
| Main model | TBD |
| Comparison | TBD |
| Demo and GitHub | TBD |
| Story and slides | TBD |

## AI use

We used an AI coding assistant (Claude Code, by Anthropic) throughout the project. It helped write the code for the
data pipeline, the models, the evaluation and the demo shop; it ran an independent review of our data rules before
training and checked the evaluation for test-set leakage; it mapped the project to the course slides and drafted the
project log, this README and the report.

What the team did and checked: every decision about the problem, data, label, models and evaluation is ours and is
recorded with its reasons in `CLAUDE.md` and `docs/PROJECT_LOG.md`. The data rules are covered by tests in `tests/`.
We trained every model on our own data; no pre-trained model is used (the Transformer reuses only the course workshop
code). Every number in the report comes from the files in `results/`. Every team member reviewed the code and can
explain how it works.

## Live audience board (presentation)

Audience members open the hosted shop on their phones (QR code); the projector shows the presenter board.

- **Open the board:** `<app url>/?board=1` (it refreshes itself every 2 seconds and shows the QR code to join).
- **Presenter menu** (hide a shopper by nickname, reset the board): `<app url>/?board=1&key=<board_key>`. The key is
  `board_key` in Streamlit secrets (Cloud: app menu, Settings, Secrets; local: copy `.streamlit/secrets.toml.example`
  to `.streamlit/secrets.toml`, which is git-ignored). If no key is set the fallback is `mercato`: set your own.
- **Rehearse alone:** add `&demo=12` (only works with the key) to fill the board with fake shoppers; Reset clears them.
- The QR code points to the app's own address; set `join_url` in secrets to override it. Shopper data lives only in
  server memory (nickname, chances, cart value), expires after 15 minutes idle, max 200 sessions.
- Tests: `.venv/bin/python tests/test_live.py` and `.venv/bin/python tests/test_app.py`.
