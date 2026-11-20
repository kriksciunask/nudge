# What a shop needs to track for Nudge

Nudge learns from a shop's own click log. This page lists what the shop must record so Nudge can work,
and what it can add so Nudge gets more accurate. Event names in brackets are the matching
Google Analytics 4 recommended e-commerce events, which many shop builders can already send.

## 1. Required: Nudge works with only these

One row per event:

| Field | Example | Why Nudge needs it |
| --- | --- | --- |
| Event time | `2019-10-06 16:55:58 UTC` | Order of clicks, time between clicks, train/test split by date |
| Event type: product view (`view_item`) | `view` | The main click word |
| Event type: add to cart (`add_to_cart`) | `cart` | Strongest single buying signal (48.6% of visits with a cart buy) |
| Event type: purchase (`purchase`) | `purchase` | The label: did this visit end in a purchase? Never shown to the model as input |
| Product id | `5800299` | The `again` signal: same product looked at twice |
| Price | `172.46` | Price band (cheap / mid / expensive, set per shop) and cart value |
| Shopper id | logged-in customer id or first-party cookie id | `new` / `returning`, and a fair train/test split by shopper |
| Visit (session) id | `0000491c-…` | Groups clicks into one visit |

That is exactly what our training data (REES46) has, so this level is proven.

## 2. Recommended: makes Nudge more accurate

| Field | Why it helps | Evidence from our project |
| --- | --- | --- |
| Remove from cart (`remove_from_cart`) | "Added then removed" means changed their mind; order matters here | Our main data has none; the cosmetics shop has it (second-shop test) |
| Checkout started (`begin_checkout`) | The step closest to buying; abandoning here is the most valuable moment to nudge | Not in our data |
| Discount or coupon used on the order | Needed to tell "bought anyway" from "bought because of the discount" | Not in our data; without it we can only predict, not prove the discount caused the sale |
| Messages sent and what happened (opened, clicked, bought) | Lets Nudge learn which message works for whom, not only who to message | REES46 direct-messaging data has this for another shop |
| Time actually spent on the page (or page hidden / tab switched) | Separates "reading carefully" from "walked away"; our gaps can't | Our longest in-visit gap was 333 hours; we had to guess with a 30-minute cut-off |
| Category | Shops with many categories: browsing across vs inside one category | 33% missing in our data, so we left it out |
| Device (mobile / desktop) | People browse on mobile and buy on desktop | Not in our data; shows up as failure #3 (bought on another device) |
| Traffic source (ad, email, search, direct) | Someone arriving from a reminder email behaves differently | Not in our data |
| Logged in or guest | Guests cannot be messaged, and their history is shorter | Our demo assumes everyone is logged in |
| Search terms | A precise search ("iPhone 15 128GB") shows clearer intent than browsing | Not in our data |

## 3. Needed to prove Nudge earns money

- **A random hold-out group.** A small random share of shoppers who would get a message get none.
  Comparing them with the rest shows how many sales the message actually caused (an A/B test).
  Prediction alone cannot show this. This is the first thing we would ask Hostinger to run.

## 4. Privacy and consent

- Use a pseudonymous shopper id (no names or emails in the click log).
- Record **marketing consent per customer** (needed for WhatsApp and email under GDPR). Nudge only
  decides; the shop sends a message only to customers who opted in.
- Keep the click log only as long as needed for retraining (for example, the last 3 months).
