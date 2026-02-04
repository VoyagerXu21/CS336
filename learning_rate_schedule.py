# learning_rate_schedule.py
from __future__ import annotations

import math


def lr_cosine_with_warmup(
    t: int,
    alpha_max: float,
    alpha_min: float,
    Tw: int,
    Tc: int,
) -> float:
    """
    Cosine learning rate schedule with warmup (as defined in the assignment).

    (Warm-up)        if t < Tw:
        alpha_t = (t / Tw) * alpha_max

    (Cosine anneal)  if Tw <= t <= Tc:
        alpha_t = alpha_min + 0.5 * (1 + cos(((t - Tw)/(Tc - Tw)) * pi)) * (alpha_max - alpha_min)

    (Post-anneal)    if t > Tc:
        alpha_t = alpha_min
    """
    # ---------- basic checks ----------
    if not isinstance(t, int):
        raise TypeError(f"t must be int, got {type(t)}")
    if not isinstance(Tw, int) or not isinstance(Tc, int):
        raise TypeError(f"Tw and Tc must be int, got Tw={type(Tw)}, Tc={type(Tc)}")
    if t < 0:
        raise ValueError(f"t must be non-negative, got {t}")
    if Tw < 0 or Tc < 0:
        raise ValueError(f"Tw and Tc must be non-negative, got Tw={Tw}, Tc={Tc}")
    if Tc < Tw:
        raise ValueError(f"Require Tc >= Tw, got Tc={Tc}, Tw={Tw}")

    amax = float(alpha_max)
    amin = float(alpha_min)

    # ---------- 1) warmup ----------
    # Tw == 0 means "no warmup"
    if Tw > 0 and t < Tw:
        return (t / Tw) * amax

    # ---------- 2) cosine anneal ----------
    if t <= Tc:
        # Edge case: Tc == Tw => cosine span length = 0 (avoid divide-by-zero)
        if Tc == Tw:
            # At t==Tw, cosine argument is 0 => lr should be alpha_max
            return amax

        progress = (t - Tw) / (Tc - Tw)  # in [0, 1]
        cosine = 0.5 * (1.0 + math.cos(progress * math.pi))  # in [0, 1]
        return amin + cosine * (amax - amin)

    # ---------- 3) post-anneal ----------
    return amin

