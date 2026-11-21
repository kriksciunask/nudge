"""Decision layer: turns the model's probability into one of three actions.

The shop owner moves one slider, "how generous are you with discounts" (0 to 1).
It moves two cut-offs:
    p < low          -> nothing            (just browsing; a message would annoy)
    low <= p < high  -> reminder + 10%     (hesitant buyer; this is where a nudge pays)
    p >= high        -> reminder only      (would buy anyway; a discount wastes margin)

More generous = wider discount band = more buyers caught (recall up),
but more discounts go to people who would have bought anyway (precision down).
"""

NOTHING = "nothing"
DISCOUNT = "reminder_discount"
REMINDER = "reminder"

LABELS = {
    NOTHING: "No message",
    DISCOUNT: "Reminder + 10% discount",
    REMINDER: "Reminder only (no discount)",
}

DISCOUNT_RATE = 0.10


def cutoffs(generosity: float) -> tuple[float, float]:
    """Slider value 0..1 -> (low, high) cut-offs on the model's probability.

    Set from the trained Transformer's scores on VALIDATION shoppers (21 Nov). The scores are well calibrated
    (a score of 0.2 means about 20% really come back), half of all visits score under 0.07, half of the cart
    abandoners under 0.19, and only ~3% of cart abandoners score over 0.4.
      stingy (0):    no message under 8%,  discount 8-20%,  reminder only over 20%
      middle (0.5):  no message under 5.5%, discount 5.5-30%, reminder only over 30%
      generous (1):  no message under 3%,  discount 3-40%,  reminder only over 40%
    """
    low = 0.08 - 0.05 * generosity
    high = 0.20 + 0.20 * generosity
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
