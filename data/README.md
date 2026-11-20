# Data

Raw data is NOT in this repo (too big). Heavy work runs in Kaggle Notebooks with the dataset attached.

| Dataset | Use | Link |
| --- | --- | --- |
| REES46 multi-category store | Main model (sessions → buy / not buy) | https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store |
| REES46 cosmetics shop | Backup / second-shop test | https://www.kaggle.com/datasets/mkechinov/ecommerce-events-history-in-cosmetics-shop |
| UCI Online Shoppers Purchasing Intention | Day-1 test of baseline/evaluation code | https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset |
| REES46 direct messaging (messages-demo.csv) | Optional model 2 | https://www.kaggle.com/datasets/mkechinov/direct-messaging |
| Hillstrom email experiment | Optional uplift | https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html |

## Rules
- Only small samples (< 50 MB) go in `data/sample/`. Everything else stays on Kaggle or in `data/raw/` (ignored by git).
- How a training example is built: group events by session, sort by time, turn each event into a token
  (`view`, `cart`, `remove`), cut just before the first purchase, label 1 if a purchase followed, else 0.
- Train/test split by user or by date, never randomly by row.
