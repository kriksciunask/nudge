# Nudge – project brief for Claude Code

This file holds the whole project context. Claude Code reads it automatically when it sits in the project folder. Keep it updated as decisions change (see "Current status" at the bottom).

## Who you are working with

- Kristupas, Economics and Data Analytics student at ISM University (Vilnius), on a Mac. He is one of 5 group members.
- His Python goes up to loops and if-statements. The code will be written with AI help ("vibe-coded"); he focuses on theory and must be able to explain everything to the lecturer.
- How to help:
  - Explain code block by block, not line by line: what the block does, which course concept or formula it implements, and what its output means. Skip syntax unless asked.
  - Plain, short explanations. One step at a time; wait for his answer before big next steps.
  - After writing code for him, ask 1–2 quick questions so he could defend it (e.g. predict a shape, say what changes if a setting changes).
  - Ask before anything that touches GitHub (creating repos, inviting people, pushing) or installs software.
  - Remind the group to acknowledge AI use in the submission.

## The course and the rules

- Course: IT102 Artificial Intelligence and Machine Learning, ISM University, lecturer Dr. Simonas Čepėnas. Textbook: Dive into Deep Learning (d2l.ai).
- Group project = 40% of the final grade. The group has 5 people.
- Requirements (Week 8 slides):
  - Build one of the models discussed in the course: linear model, MLP, CNN, RNN, Transformer. If you choose a Transformer, add an alternative specification as well.
  - It must solve a real problem.
  - Deploy the model via GitHub.
  - Submission of project and presentation: 4 December 2026, 23:00. Presentation: 10 December 2026 (preliminary), Cyber City. The audience includes Hostinger executives.
  - A downloaded Hugging Face model may only be a baseline or comparison, never the main model.
- Grading rubric (100 points):
  - Idea 25: a real problem, who has it, why it matters; the model fits the problem and we can say why.
  - Execution 40: model built, trained and tested correctly; compared with a simple baseline; five failures shown and explained.
  - Live demo 10: runs from GitHub; works on input from the audience or commission.
  - Presentation 25: clear to a non-specialist; every member can explain it and answer questions.

## Where the idea came from

- We present to Hostinger. Kristupas's friend Lukas Jablonskas works there. Lukas said:
  - Most Hostinger websites are e-shops and small-business pages.
  - Companies often don't want to block AI scrapers, because those let them be found through AI (so we dropped the "AI crawler detector" idea).
  - Suggested direction: an AI agent that Hostinger users add to their e-shop; extra mile: India and South-East Asia, where e-commerce happens through WhatsApp, so an agent in the company's WhatsApp.
- Kristupas narrowed it to: follow-ups after a shopper views or adds to cart and leaves ("you looked at this, here's 10%"), but smarter than today's automatic flows.
- Hostinger already has Hostinger Reach with an abandoned-cart flow for its store builder. It is email-based, checks abandoned carts every hour, and treats every cart the same. Nudge is the smart decision layer that is missing.

## The idea: Nudge

- Problem: about 70% of online carts are abandoned (Baymard: 70.22%, average of 50 studies), and many of those people were only browsing (42% of US shoppers abandoned a cart because they were "just browsing / not ready to buy"). Shops send the same reminder, often with a discount, to everyone. So they give margin away to people who would have bought anyway, annoy people who were only browsing, and teach customers to abandon carts on purpose to get a coupon.
- What Nudge does: a sequence model reads the session's clicks in order and outputs the chance the person buys without help. A decision layer turns that into one of three actions:
  - High chance: no discount (it would be wasted margin); at most a reminder.
  - Middle: reminder + 10% discount. This is where a nudge pays off.
  - Very low (just browsing): nothing.
- The business controls one setting, a "how generous are you with discounts" slider, which moves the cut-offs. This is the precision/recall trade-off from the course: low cut-off catches more buyers but gives discounts to people who would buy anyway (more false positives); high cut-off saves margin but misses some buyers (more false negatives).
- Money logic example: €50 cart, 10% = €5. If someone had an 80% chance of buying anyway, the code wastes on average 0.8 × €5 = €4.
- What we do NOT build: the sending workflow itself (email/WhatsApp delivery, scheduling). Nudge is the decision layer on top of tools like Hostinger Reach.
- Scope: the model is trained on all sessions (browse-only and cart sessions), so it covers both browse and cart abandonment. In production, only customers with a phone number and opt-in can get WhatsApp messages; in our demo every visitor is a logged-in customer.
- Name: project "Nudge"; report title "Predicting Purchase Intent".
- Strong pitch argument (economics / game theory): blanket discount flows train shoppers to abandon carts for coupons; Nudge breaks that loop because a discount hunter's behaviour looks different from a hesitant real buyer's.

## Models

- Why a sequence model: a session is a sequence of clicks, like a word is a sequence of letters. Order matters: "cart → remove" (changed their mind) and "remove → cart" have the same counts but opposite meaning. Sessions have different lengths.
- Main model: Transformer encoder. Each click attends to all other clicks (the attention function from the course workshop: softmax(QKᵀ/√d)V); position embedding keeps the order; output = one probability per session (sigmoid). Attention heatmap shows which click drove the decision (good for the presentation).
- Alternative specification: RNN (GRU or LSTM). Reads clicks one at a time and keeps a hidden state: h_new = tanh(W·x + U·h_old + b); final hidden state → sigmoid → probability. Weakness: early clicks fade in long sessions (vanishing gradients).
- Baselines:
  1. Rule: "discount everyone who adds to cart" (what shops do today).
  2. Logistic regression on counts (number of views, carts, removes, cart value). Ignores order.
- Story of the comparison: does order matter (RNN vs counts)? Do long sessions matter (Transformer vs RNN, compare on short vs long sessions)?
- Reuse: the course workshop (Part 1, name generator) has a working small Transformer in PyTorch (NameGPT: attention, causal_mask, SelfAttention, Block) and a Streamlit app.py. Kristupas has these files; reuse the structure, but change the task from "predict the next letter" to "one yes/no per session". For whole-session classification a causal mask is not required.

## Evaluation

- Test set split by user or by date, never randomly by row (otherwise information leaks).
- Strong class imbalance (most sessions don't buy): report precision, recall, F1 and a confusion matrix, not accuracy alone. Consider class weights or resampling.
- Profit comparison: Nudge at several cut-offs vs "discount everyone".
- Five failures to show (collect real test-set examples):
  1. One-click sessions: too little information.
  2. Returning customers: the model only sees this session.
  3. Bought later on another device: labelled as a non-buyer.
  4. Very long, indecisive sessions.
  5. Expensive products, where hesitation looks different.
- Honest limitation for Q&A: Nudge predicts who buys; proving that the discount itself causes extra sales needs an A/B test (some people randomly get the discount). Propose that Hostinger runs it as the next step.

## Data

Decided 20 November: model 1 is trained on the multi-category store.

| Dataset | Use | Notes |
| --- | --- | --- |
| REES46 eCommerce behavior data from multi-category store – https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store | MAIN: model 1 | One row per event (event_time, event_type, product_id, category, brand, price, user_id, user_session). About 15 GB: work in a Kaggle Notebook with the dataset attached, take one month, sample (e.g. 50k sessions). Check on day 1 whether remove_from_cart exists; if not, add price band and category to each click token. |
| REES46 eCommerce Events History in Cosmetics Shop – https://www.kaggle.com/datasets/mkechinov/ecommerce-events-history-in-cosmetics-shop | Backup / second-shop test | Same format, Oct 2019–Feb 2020, includes remove_from_cart, about 2 GB. Test: does the trained model still work on a different shop? |
| Coveo SIGIR eCom 2021 – https://github.com/coveooss/SIGIR-ecom-data-challenge | Backup | Session data with a cart-abandonment task (predict purchase after add-to-cart). Sign-up, research/educational use only. |
| UCI Online Shoppers Purchasing Intention – https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset | Day-1 test of baseline/evaluation code | 12,330 sessions, 17 summary features, 15.5% buy. No click order, so not for the sequence model. |
| REES46 E-commerce multichannel direct messaging 2021–2023 – https://www.kaggle.com/datasets/mkechinov/direct-messaging | Optional model 2 | Email, web push, mobile push, SMS messages of a medium-size store; bulk, trigger (e.g. abandoned cart) and transactional campaigns; opened, clicked, purchase per message. Use messages-demo.csv (10M rows), not the 21.5 GB full file. No WhatsApp, not randomised. Check for a discount field. |
| Hillstrom email experiment – https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html | Optional uplift | 64,000 customers randomly split: men's email, women's email, no email. Shows whether a message causes sales. |

Building training data: group events by session, sort by time, turn each event into a token, cut the session just before the first purchase, label 1 if a purchase followed, else 0.

## Optional model 2 (decide on 27 November)

- Question it answers: which channel and when to send. Trained on the direct-messaging data (e.g. predict whether a trigger message gets opened/clicked/converts from channel, hour sent, customer history). An MLP is enough.
- Only if model 1 is on track by 27 November. Otherwise offer type, timing (e.g. 2 h, then 24 h) and message wording stay simple rules, presented as "what we'd train next with Hostinger's data".
- Caveat to say openly: model 2 is trained on different data than model 1 and only connected in the demo.

## The shared contract

- A session is a Python list of event strings in time order, e.g. ["view", "view", "cart", "remove"]. Vocabulary: "view", "cart", "remove" (+ optional price band / category tokens later). Purchase is never part of the input.
- One function connects model and app: predict(session: list[str]) -> float (chance to buy, 0 to 1).
- Until the real model exists, predict is a clearly marked placeholder so the app can be built in parallel.
- Implemented in `nudge/contract.py` (vocabulary, `encode`, `predict`). The decision layer (slider → cut-offs → action) is in `nudge/decision.py`.

## Demo

- Runs in our own controlled setup (not inside Hostinger), from the GitHub repo, with Streamlit (same style as the workshop app.py).
- A fake electronics shop with about 6 products (matching the training data). A commission member is the "logged-in customer": views, adds to cart, removes.
- The click sequence is shown live. "Leave the shop" → predict() → probability, the decision (no message / reminder / reminder + 10%), and a WhatsApp-style message preview.
- A "how generous are you with discounts" slider moves the cut-offs; a chart shows profit vs "discount everyone".
- "Look inside": attention heatmap of the Transformer.
- Optional if reliable by 1 December: send the message for real to a group member's phone. Otherwise the preview is enough. In the pitch, one line: production needs customer opt-in and the WhatsApp Business API.
- Keep a recorded backup video of the demo. Test with people outside the group before 10 December.

## Repo structure

```
nudge/  (repo root)
  README.md            what it is, how to run, team, AI-use acknowledgement
  CLAUDE.md            this file
  requirements.txt     first line: --extra-index-url https://download.pytorch.org/whl/cpu ; then torch, streamlit, pandas, numpy, scikit-learn, matplotlib, altair
  .gitignore           data/raw/, *.csv, *.parquet, *.zip, *.gz, .venv/, __pycache__/, .ipynb_checkpoints/, .DS_Store (keep app/model.pt)
  data/README.md       where the data comes from; only small samples (<50 MB) in data/sample/
  nudge/contract.py    event vocabulary, session → token ids, predict()
  nudge/decision.py    slider → cut-offs → action (no message / reminder / reminder + 10%)
  notebooks/           01_data, 02_baselines, 03_rnn, 04_transformer, 05_evaluation (each starts with a plain-words markdown cell: purpose, input, output, owner)
  app/app.py           Streamlit demo
  app/model.pt         trained model
```

- Heavy data work and training: Kaggle Notebooks (dataset attached directly, free GPU). Save the notebooks into the repo.
- Never commit data files. Commit and push at the end of every working session.
- Local Python: use 3.12 or 3.13 (PyTorch may not support 3.15 yet). On Kristupas's Mac: `python3.13`.

## Team roles (names to fill in)

| Role | Owns |
| --- | --- |
| Data | sample the dataset, build sequences and labels, train/test split |
| Main model | Transformer training and tuning, attention heatmap |
| Comparison | RNN, both baselines, metrics, the five failures |
| Demo and GitHub | Streamlit shop, slider, WhatsApp preview, repo, README |
| Story and slides | business case, Hostinger and India/SEA research, profit numbers, slides, report |

Every member must be able to explain the whole model. Around 2 December each person explains their part to the others in 5 minutes.

## Timeline

- 20–23 Nov: data + check with Lukas. First checkpoint Sunday evening: data sample saved, baseline running on UCI, app shell clickable.
- 24–28 Nov: models and baselines. 27 Nov: decide on model 2.
- 29 Nov–1 Dec: demo and the five failures.
- 2–4 Dec: slides, report, rehearsal. 4 Dec 23:00: submission.
- 10 Dec: presentation.
- If data isn't ready by Monday 23 Nov, take a smaller sample rather than slipping the model week.

## Facts for the pitch (with sources)

- Average cart abandonment 70.22% (50 studies) – Baymard Institute, https://baymard.com/lists/cart-abandonment-rate
- 42% of US shoppers abandoned a cart because they were "just browsing / not ready to buy" – Baymard (same page)
- Hostinger Reach abandoned-cart flow is email-based and checks every hour – https://www.hostinger.com/support/?p=8204
- Indian SMEs surveyed: 93% sell on WhatsApp, 21% on a platform like Flipkart – Peak XV, https://peakxv.com/?p=555
- About 40% of Indonesian e-commerce transactions are informal (WhatsApp, Instagram, Facebook), per McKinsey – https://www.xendit.co/en/blog/sell-online-easily-with-chat-commerce/
- India: email open rates around 15–20%; WhatsApp is the default channel for 500M+ users – https://www.hillteck.com/blog/recover-abandoned-carts-whatsapp-shopify.html
- Counterpoint: once low opt-in coverage is counted, WhatsApp and email both recover roughly 1% of all abandoned carts – https://dev.to/marcus_the_architect/whatsapp-vs-email-cart-recovery-rates-costs-setup-2026-7j6
- Treat WhatsApp vendor blogs claiming 15–60% recovery as marketing, not evidence.

## Risks

- Prediction ≠ causation (see Evaluation).
- Data size: sample early.
- Demo fragility: backup video.
- Privacy: WhatsApp marketing in the EU needs opt-in (GDPR). Mention, don't build.

## Open questions

- [ ] Lukas: does Hostinger care about this version? (send him a one-line summary)
- [ ] Model 2 or rules? Decide 27 Nov.
- [ ] Demo hosting: Streamlit Community Cloud, or run from GitHub on a laptop?
- [ ] Does the multi-category data have remove_from_cart events?
- [ ] Teammates' GitHub usernames (for repo invites).

## Current status (update this section as you go)

- 20 Nov: idea, datasets and plan decided. Shared plan doc: https://claude.ai/artifact/34NsQF3WLAiJPVRyCwgVV4
- 20 Nov: local repo structure created (README, CLAUDE.md, .gitignore, requirements.txt, data/README.md, nudge/contract.py with PLACEHOLDER predict, nudge/decision.py, 5 notebook stubs, app/app.py demo shell).
- 20 Nov: private repo https://github.com/kriksciunask/nudge created and pushed. Local .venv (Python 3.13, torch 2.15 CPU, streamlit 1.67) installed; app tested (opens, click-through works).
- NEXT:
  1. Teammates accept the GitHub invites (sent 20 Nov, write access: ulapask, gusteruzgaite, jokubasspir-ops, bukauskaiteliepa-glitch).
  2. Set up the Kaggle notebook for 01_data (first check: does remove_from_cart exist?).
