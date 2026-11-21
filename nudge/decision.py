"""Decision layer: turns the model's probability into one of three actions.

The shop owner moves one slider, "how generous are you with discounts" (0 to 1).
It moves two cut-offs:
    p < low          -> nothing            (just browsing; a message would annoy)
    low <= p < high  -> reminder + 10%     (hesitant buyer; this is where a nudge pays)
    p >= high        -> reminder only      (would buy anyway; a discount wastes margin)

More generous = wider discount band = more buyers caught (recall up),
but more discounts go to people who would have bought anyway (precision down).
"""

import numpy as np

NOTHING = "nothing"
DISCOUNT = "reminder_discount"
REMINDER = "reminder"

LABELS = {
    NOTHING: "No message",
    DISCOUNT: "Reminder + 10% discount",
    REMINDER: "Reminder only (no discount)",
}

DISCOUNT_RATE = 0.10


# 40th, 50th, 60th, 70th, 80th percentile of the final Transformer's scores on the VALIDATION cart abandoners
# (average of the 3 seeds, results/tuned/transformer_4eb04ce3_s{42,43,44}_val.npy; 21 Nov). Test is never used here.
HIGH_AT_PERCENTILE = (0.1745, 0.1943, 0.2146, 0.2378, 0.2662)
GENEROSITY_AT = (0.0, 0.25, 0.5, 0.75, 1.0)


def cutoffs(generosity: float) -> tuple[float, float]:
    """Slider value 0..1 -> (low, high) cut-offs on the model's probability.

    HIGH ("would buy anyway, no discount above this") is a percentile of the validation scores of cart abandoners,
    the same rule 05_evaluation used for "Transformer (60%)": discount the 60% of cart abandoners with the lowest
    scores. The slider spreads it from the 40th percentile (g=0, stingy: ~0.17) over the 60th (g=0.5, middle: 0.215,
    the setting we report) to the 80th (g=1, generous: ~0.27). Chosen on validation only, never on test.
    This is the course's precision/recall trade-off (Week 3): a cut-off picked on validation data, then moved
    to catch more buyers (generous) or to waste fewer discounts (stingy).

    LOW ("just browsing, send nothing") stays a fixed 8% (g=0) down to 3% (g=1). It is an addition to the
    reported "@60%" policy (which has no lower limit), so the money at g=0.5 differs slightly from the
    reported numbers (see scripts/make_money_curve.py).
    """
    low = 0.08 - 0.05 * generosity
    high = float(np.interp(generosity, GENEROSITY_AT, HIGH_AT_PERCENTILE))
    return low, high


def decide(p: float, generosity: float) -> str:
    low, high = cutoffs(generosity)
    if p < low:
        return NOTHING
    if p < high:
        return DISCOUNT
    return REMINDER


def expected_wasted_discount(p: float, cart_value: float) -> float:
    """If we give a discount, on average p of it goes to someone who would have bought anyway."""
    return p * DISCOUNT_RATE * cart_value
