# Baselines (test shoppers, never seen in training)

Visits: train 933,419, valid 115,544, test 117,531. Label = comes back and buys within 7 days (8.6% of test visits). 95% ranges from resampling shoppers 200 times.

## Prediction quality

| Model | AUC | PR-AUC | Precision | Recall | F1 | AUC on one-click visits |
|---|---|---|---|---|---|---|
| Random | 0.501 (0.496–0.507) | 0.086 (0.082–0.090) | 0.086 | 0.507 | 0.148 (0.142–0.154) | 0.495 |
| Industry rule (cart = 1) | 0.564 (0.557–0.571) | 0.124 (0.119–0.130) | 0.208 | 0.184 | 0.196 (0.186–0.204) | 0.507 |
| Logistic regression | 0.675 (0.668–0.681) | 0.159 (0.151–0.167) | 0.166 | 0.349 | 0.225 (0.215–0.234) | 0.657 |

## Money on cart abandoners (10% discount, max one per shopper per week)

| Policy | Discounts sent | € given to shoppers who came back anyway | Spared a discount and came back | Left without a nudge |
|---|---|---|---|---|
| Industry rule (everyone) (100% of cart abandoners) | 6,622 | €50,327 (€45,280–€56,689) | 651 | 1,568 (1,461–1,680) |
| Logistic regression (80% of cart abandoners) | 5,742 | €39,825 (€35,920–€45,048) | 868 | 2,231 (2,097–2,364) |
| Random (80% of cart abandoners) | 5,524 | €43,255 (€38,781–€49,638) | 828 | 2,489 (2,355–2,617) |
| Logistic regression (60% of cart abandoners) | 4,625 | €30,465 (€26,927–€35,080) | 1,088 | 3,128 (2,976–3,275) |
| Random (60% of cart abandoners) | 4,353 | €35,678 (€31,346–€41,086) | 1,027 | 3,461 (3,321–3,603) |
| Logistic regression (40% of cart abandoners) | 3,365 | €21,699 (€18,503–€25,942) | 1,327 | 4,149 (3,981–4,305) |
| Random (40% of cart abandoners) | 3,111 | €28,046 (€23,864–€32,805) | 1,235 | 4,495 (4,341–4,657) |

## What logistic regression learned (words used 200+ times)

Raise the come-back chance most: `cart_mid_again_long` (+0.35), `cart_expensive_again_normal` (+0.31), `view_expensive_away` (+0.29), `cart_expensive_again_quick` (+0.28), `cart_mid_again_normal` (+0.28), `cart_cheap_again_normal` (+0.27)

Lower it most: `new` (-1.41), `view_expensive_start` (-0.57), `view_mid_start` (-0.52), `view_cheap_start` (-0.51), `returning` (-0.13), `view_expensive_normal` (-0.00)
