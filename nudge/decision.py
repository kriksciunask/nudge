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
    """Slider value 0..1 -> (low, high) cut-offs. 0 = band 0.15-0.45, 1 = band 0.05-0.80."""
    low = 0.15 - 0.10 * generosity
    high = 0.45 + 0.35 * generosity
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
