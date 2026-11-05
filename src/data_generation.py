"""Synthetic credit-card transaction generator with ~1% injected fraud.

Why synthetic?
--------------
Public fraud datasets are either PCA-anonymised (no interpretable features),
behind a login wall, or licence-restricted. A documented generative model lets
us (a) ship a fully reproducible project with no external downloads and
(b) know exactly which behavioural signals separate fraud from normal traffic.

Generative model (summary — full description in README / paper §3.1)
---------------------------------------------------------------------
1. **Customers.** Each customer gets a spending multiplier, a personal
   preference over 10 merchant categories, a daily activity rate, a
   chronotype (preferred hour shift), an online-shopping propensity and,
   with some probability, one trip away from home.
2. **Legitimate transactions.** A Poisson number of purchases per customer
   over ``N_DAYS``; each purchase samples category, hour, amount, channel
   (card-present vs. online) and geolocation distance from the customer's
   own profile. Noise is deliberate: legitimate customers also travel, buy
   electronics, shop at night, etc.
3. **Fraud.** Compromise *events* hit random customers at random times. An
   event emits a *burst* of fraudulent transactions whose style depends on
   the scenario:

   ============  =====  ========================================================
   scenario      share  behaviour
   ============  =====  ========================================================
   cnp_spree      45%   online, risky categories, 2-6x usual amount, far IP
   card_testing   20%   many tiny online charges (<$5) minutes apart
   stolen_card    20%   card-present, far from home, high amounts, often night
   mimic          15%   copies the customer's own habits (hard / near-invisible)
   ============  =====  ========================================================

   Events are added until the fraud share reaches ``TARGET_FRAUD_RATE``.

The generator only emits *raw* observable fields (time, amount, category,
channel, distance). Behavioural features (velocity, deviation from the
customer's history, ...) are derived causally in ``src/features.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

# --------------------------------------------------------------------------
# Merchant-category catalogue
#   popularity : base share of legitimate purchases
#   median     : median ticket size in USD for an average customer
#   p_online   : probability a legitimate purchase in this category is online
# --------------------------------------------------------------------------
CATEGORIES = {
    #  name               popularity median p_online
    "grocery":            (0.22,   45.0, 0.04),
    "restaurant":         (0.17,   30.0, 0.06),
    "gas_transport":      (0.14,   40.0, 0.03),
    "retail_apparel":     (0.09,   60.0, 0.25),
    "online_retail":      (0.12,   50.0, 0.95),
    "electronics":        (0.03,  150.0, 0.40),
    "travel":             (0.03,  300.0, 0.70),
    "entertainment":      (0.08,   35.0, 0.30),
    "health":             (0.08,   40.0, 0.05),
    "gift_cards_digital": (0.04,   40.0, 0.80),
}
CAT_NAMES = list(CATEGORIES)
CAT_POP = np.array([v[0] for v in CATEGORIES.values()])
CAT_POP = CAT_POP / CAT_POP.sum()
CAT_MEDIAN = np.array([v[1] for v in CATEGORIES.values()])
CAT_P_ONLINE = np.array([v[2] for v in CATEGORIES.values()])
CAT_IDX = {c: i for i, c in enumerate(CAT_NAMES)}

# Categories fraudsters favour (easy to monetise / resell).
RISKY_CNP = ["electronics", "gift_cards_digital", "online_retail", "travel"]
RISKY_CP = ["electronics", "retail_apparel", "gas_transport", "grocery"]
TESTING_CATS = ["gift_cards_digital", "online_retail", "entertainment"]

FRAUD_SCENARIOS = ["cnp_spree", "card_testing", "stolen_card", "mimic"]
FRAUD_SCENARIO_P = [0.45, 0.20, 0.20, 0.15]

SECONDS_PER_DAY = 86_400
START_DATE = pd.Timestamp("2025-01-01")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _legit_hours(rng: np.random.Generator, n: int, shift: float) -> np.ndarray:
    """Hour-of-day for legitimate purchases: mostly daytime, some uniform."""
    day = rng.normal(14.0 + shift, 3.5, n)
    anytime = rng.uniform(0, 24, n)
    hours = np.where(rng.random(n) < 0.88, day, anytime)
    return np.mod(hours, 24.0)


def _fraud_start_hour(rng: np.random.Generator) -> float:
    """Fraud bursts start disproportionately at night (00:00-05:59)."""
    return rng.uniform(0, 6) if rng.random() < 0.55 else rng.uniform(0, 24)


def _make_customers(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Sample static customer profiles."""
    prefs = rng.dirichlet(CAT_POP * 12.0, size=n)          # personal category mix
    trip = rng.random(n) < 0.15                            # takes one trip?
    trip_start = rng.uniform(0, config.N_DAYS - 10, n)
    trip_len = rng.uniform(3, 10, n)
    return pd.DataFrame({
        "customer_id": np.arange(n),
        "spend_mult": rng.lognormal(0.0, 0.5, n),          # wealth / spend level
        "rate_per_day": rng.lognormal(np.log(0.75), 0.5, n),
        "hour_shift": rng.normal(0.0, 1.5, n),             # chronotype
        "online_mult": rng.lognormal(0.0, 0.4, n),         # online propensity
        "home_noise_km": rng.lognormal(np.log(5.0), 0.4, n),
        "trip": trip,
        "trip_start": np.where(trip, trip_start, -1.0),
        "trip_end": np.where(trip, trip_start + trip_len, -1.0),
        "trip_dist": rng.uniform(200, 3_000, n),
        "prefs": list(prefs),
    })


def _legit_transactions(rng: np.random.Generator, cust: pd.DataFrame) -> pd.DataFrame:
    """Generate all legitimate transactions for every customer."""
    frames = []
    for c in cust.itertuples(index=False):
        n = rng.poisson(c.rate_per_day * config.N_DAYS)
        if n == 0:
            continue
        day = np.floor(rng.uniform(0, config.N_DAYS, n))
        hour = _legit_hours(rng, n, c.hour_shift)
        t = day * SECONDS_PER_DAY + hour * 3_600
        cat = rng.choice(len(CAT_NAMES), size=n, p=c.prefs)
        amount = CAT_MEDIAN[cat] * c.spend_mult * rng.lognormal(0.0, 0.6, n)
        online = rng.random(n) < np.clip(CAT_P_ONLINE[cat] * c.online_mult, 0, 0.98)
        # Geolocation distance from home: card-present ~ a few km; online is the
        # IP-geolocation distance, noisier; trips put card-present txns far away.
        dist_cp = rng.lognormal(np.log(c.home_noise_km), 0.8, n)
        dist_on = rng.lognormal(np.log(15.0), 1.1, n)
        dist = np.where(online, dist_on, dist_cp)
        if c.trip:
            on_trip = (t >= c.trip_start * SECONDS_PER_DAY) & (t <= c.trip_end * SECONDS_PER_DAY)
            dist = np.where(on_trip & ~online,
                            c.trip_dist * rng.lognormal(0, 0.15, n), dist)
        frames.append(pd.DataFrame({
            "customer_id": c.customer_id, "t": t, "category_idx": cat,
            "amount": amount, "is_online": online.astype(int),
            "distance_km": dist, "is_fraud": 0, "fraud_scenario": "none",
        }))
    return pd.concat(frames, ignore_index=True)


def _fraud_event(rng: np.random.Generator, c, scenario: str) -> pd.DataFrame:
    """Generate one burst of fraudulent transactions on customer ``c``."""
    day = np.floor(rng.uniform(0, config.N_DAYS))
    t0 = day * SECONDS_PER_DAY + _fraud_start_hour(rng) * 3_600
    typical = c.spend_mult  # scales amounts relative to the victim's habits

    if scenario == "cnp_spree":
        k = 1 + min(rng.poisson(2.5), 10)
        gaps = rng.exponential(15 * 60, k)                     # ~15 min apart
        cats = np.where(rng.random(k) < 0.75,
                        rng.choice([CAT_IDX[x] for x in RISKY_CNP], k),
                        rng.choice(len(CAT_NAMES), k, p=c.prefs))
        amount = CAT_MEDIAN[cats] * typical * rng.lognormal(np.log(3.0), 0.6, k)
        online = rng.random(k) < 0.9
        dist = np.where(rng.random(k) < 0.65,
                        rng.lognormal(np.log(800), 1.0, k),    # foreign IP / proxy
                        rng.lognormal(np.log(15), 1.1, k))
    elif scenario == "card_testing":
        k = 3 + min(rng.poisson(3.0), 12)
        gaps = rng.exponential(3 * 60, k)                      # ~3 min apart
        cats = rng.choice([CAT_IDX[x] for x in TESTING_CATS], k)
        amount = rng.uniform(0.5, 5.0, k)
        # occasionally the tester "cashes out" with one larger charge at the end
        if rng.random() < 0.4:
            amount[-1] = CAT_MEDIAN[cats[-1]] * typical * rng.lognormal(np.log(2.5), 0.5)
        online = np.ones(k, dtype=bool)
        dist = rng.lognormal(np.log(600), 1.2, k)
    elif scenario == "stolen_card":
        k = 1 + min(rng.poisson(1.5), 8)
        gaps = rng.exponential(45 * 60, k)                     # ~45 min apart
        cats = np.where(rng.random(k) < 0.7,
                        rng.choice([CAT_IDX[x] for x in RISKY_CP], k),
                        rng.choice(len(CAT_NAMES), k, p=c.prefs))
        amount = CAT_MEDIAN[cats] * typical * rng.lognormal(np.log(2.2), 0.6, k)
        online = np.zeros(k, dtype=bool)
        dist = rng.lognormal(np.log(120), 0.9, k)               # another city
    else:  # "mimic": behaves like the customer — intentionally hard to catch
        k = 1 + min(rng.poisson(0.8), 4)
        gaps = rng.exponential(6 * 3_600, k)                   # hours apart
        cats = rng.choice(len(CAT_NAMES), k, p=c.prefs)
        amount = CAT_MEDIAN[cats] * typical * rng.lognormal(np.log(1.3), 0.6, k)
        online = rng.random(k) < np.clip(CAT_P_ONLINE[cats] * c.online_mult, 0, 0.98)
        dist = np.where(online, rng.lognormal(np.log(15), 1.1, k),
                        rng.lognormal(np.log(c.home_noise_km * 1.5), 0.8, k))
        t0 = day * SECONDS_PER_DAY + _legit_hours(rng, 1, c.hour_shift)[0] * 3_600

    t = t0 + np.cumsum(gaps) - gaps[0]
    return pd.DataFrame({
        "customer_id": c.customer_id, "t": t, "category_idx": cats.astype(int),
        "amount": amount, "is_online": online.astype(int), "distance_km": dist,
        "is_fraud": 1, "fraud_scenario": scenario,
    })


def generate_transactions(seed: int = config.SEED,
                          n_customers: int = config.N_CUSTOMERS,
                          target_rate: float = config.TARGET_FRAUD_RATE) -> pd.DataFrame:
    """Return a time-sorted DataFrame of raw transactions with labels.

    Columns: transaction_id, customer_id, timestamp, t_seconds, amount,
    merchant_category, is_online, distance_km, is_fraud, fraud_scenario.
    """
    rng = np.random.default_rng(seed)
    cust = _make_customers(rng, n_customers)
    legit = _legit_transactions(rng, cust)

    # Number of fraud rows needed so that fraud / total == target_rate.
    n_fraud_target = int(round(target_rate * len(legit) / (1 - target_rate)))
    fraud_frames, n_fraud = [], 0
    while n_fraud < n_fraud_target:
        c = cust.iloc[rng.integers(len(cust))]
        scenario = rng.choice(FRAUD_SCENARIOS, p=FRAUD_SCENARIO_P)
        ev = _fraud_event(rng, c, scenario)
        ev = ev[ev["t"] < config.N_DAYS * SECONDS_PER_DAY]   # stay inside window
        ev = ev.iloc[: n_fraud_target - n_fraud]              # hit the target exactly
        fraud_frames.append(ev)
        n_fraud += len(ev)

    df = pd.concat([legit, *fraud_frames], ignore_index=True)
    df["amount"] = df["amount"].clip(lower=0.5).round(2)
    df["distance_km"] = df["distance_km"].round(2)
    df["merchant_category"] = np.array(CAT_NAMES)[df["category_idx"].to_numpy()]
    df = df.sort_values(["t", "customer_id"], kind="mergesort").reset_index(drop=True)
    df.insert(0, "transaction_id", np.arange(len(df)))
    df["timestamp"] = START_DATE + pd.to_timedelta(df["t"], unit="s")
    df = df.rename(columns={"t": "t_seconds"})
    cols = ["transaction_id", "customer_id", "timestamp", "t_seconds", "amount",
            "merchant_category", "is_online", "distance_km", "is_fraud",
            "fraud_scenario"]
    return df[cols]


if __name__ == "__main__":  # quick manual check: python -m src.data_generation
    d = generate_transactions()
    print(d.head())
    print(f"rows={len(d):,}  fraud rate={d.is_fraud.mean():.4%}")
    print(d.groupby("fraud_scenario").size())
