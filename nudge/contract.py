"""The shared contract between the model and the demo app.

A session is a list of event strings in time order, e.g. ["view", "view", "cart", "remove"].
Purchase is never part of the input: we predict it.

    predict(session) -> float   chance (0 to 1) that the shopper buys without help
"""

import math

# Token vocabulary. Index 0 is padding, used later to make sessions of different
# lengths the same length inside a training batch.
VOCAB = ["<pad>", "view", "cart", "remove"]
TOKEN_TO_ID = {tok: i for i, tok in enumerate(VOCAB)}
EVENTS = VOCAB[1:]  # the events a shopper can produce

# True until the trained model (app/model.pt) is plugged in. The app shows a banner while True.
IS_PLACEHOLDER = True


def encode(session: list[str]) -> list[int]:
    """Turn a session of event strings into token ids, e.g. ["view", "cart"] -> [1, 2]."""
    unknown = [e for e in session if e not in TOKEN_TO_ID or e == "<pad>"]
    if unknown:
        raise ValueError(f"Unknown events {unknown}; allowed: {EVENTS}")
    return [TOKEN_TO_ID[e] for e in session]


def predict(session: list[str]) -> float:
    """Chance (0 to 1) that this session ends in a purchase without a nudge.

    >>> PLACEHOLDER <<<  This is a hand-written rule, NOT the trained model.
    It exists only so the app can be built before the Transformer is trained.
    Replace the body with: load app/model.pt, encode(session), run the model, sigmoid.
    """
    encode(session)  # validates the events
    if not session:
        return 0.02

    # Logistic-style score: carts push the chance up, removes push it down,
    # and the last click counts extra (a crude stand-in for "order matters").
    score = -2.5
    score += 0.05 * min(session.count("view"), 20)
    score += 1.2 * session.count("cart")
    score -= 0.9 * session.count("remove")
    if session[-1] == "cart":
        score += 0.5
    elif session[-1] == "remove":
        score -= 0.5
    return 1 / (1 + math.exp(-score))
