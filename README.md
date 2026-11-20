# Nudge – Predicting Purchase Intent

IT102 Artificial Intelligence and Machine Learning, ISM University (group project).

**The problem:** about 70% of online carts are abandoned, and shops send the same reminder (often with a discount)
to everyone. That gives margin away to people who would have bought anyway and teaches shoppers to abandon carts for coupons.

**What Nudge does:** a sequence model (Transformer encoder, with an RNN as the alternative) reads a shopper's clicks
in order and predicts the chance they buy without help. A decision layer turns that into:
no message / reminder only / reminder + 10% discount. The shop owner sets one "how generous" slider.

## Run the demo

You need git and Python 3.12 or 3.13 (not 3.15: PyTorch may not support it yet).

```bash
git clone https://github.com/kriksciunask/nudge.git
cd nudge
python3.13 -m venv .venv          # or python3.12
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run app/app.py
```

The browser opens at http://localhost:8501. Next time: `cd nudge`, `source .venv/bin/activate`, `git pull`, then the last line.

## Folders

| Folder | What's in it |
| --- | --- |
| `nudge/` | `contract.py`: event vocabulary and `predict(session) -> float`; `decision.py`: slider → action |
| `notebooks/` | 01_data → 05_evaluation (run on Kaggle, saved here) |
| `app/` | Streamlit demo shop and the trained `model.pt` |
| `data/` | Where the data comes from (raw data is not committed) |

## Team

| Role | Name |
| --- | --- |
| Data | TBD |
| Main model | TBD |
| Comparison | TBD |
| Demo and GitHub | TBD |
| Story and slides | TBD |

## AI use

Parts of the code and text in this project were written with the help of AI assistants (Claude). Every team member
reviewed the code and can explain how it works.
