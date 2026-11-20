"""Simulate the users of a paywall price test.

Run from the repository root:

    python data/simulate.py

It writes data/users.csv, one row per user who saw the paywall. Every number below is
an assumption, not an observation. The seed is fixed, so the file is identical on every run.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ---- assumptions ---------------------------------------------------------------------

SEED = 20261113

# Users per group. From the sample-size calculation: detect a fall in conversion from
# 5.0% to 4.6% with a 5% significance level and 80% power (two-sided, two proportions).
N_PER_GROUP = 44_829

# The test runs for 8 weeks. Users arrive evenly over the period.
TEST_START = "2025-01-06"
TEST_DAYS = 56

# Monthly price shown on the paywall, in EUR.
PRICE = {"A": 4.99, "B": 6.99}

# Share of users who pay at least once, once any novelty effect has faded.
PAID_RATE = {"A": 0.050, "B": 0.046}

# Share of trial users who go on to pay. Assumed equal in both groups.
TRIAL_TO_PAID = 0.42

# Chance that a paying user cancels before each new month. A higher price is assumed
# to shorten subscriptions. These values give about 7.0 and 6.25 paid months on average.
MONTHLY_CHURN = {"A": 0.105, "B": 0.130}

# Novelty effect, group B only: the new paywall draws extra trial starts at first.
# The boost starts at +25% on day 0 and fades with a 5-day time constant.
NOVELTY_BOOST = 0.25
NOVELTY_DECAY_DAYS = 5.0

# Users are followed for 12 months after they see the paywall.
FOLLOW_UP_MONTHS = 12

OUT = Path(__file__).resolve().parent / "users.csv"

# ---- simulation ----------------------------------------------------------------------


def simulate(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 2 * N_PER_GROUP

    # Random assignment, exactly balanced.
    group = rng.permutation(np.repeat(["A", "B"], N_PER_GROUP))
    is_b = group == "B"

    day = rng.integers(0, TEST_DAYS, size=n)
    exposure_date = pd.Timestamp(TEST_START) + pd.to_timedelta(day, unit="D")

    # Trial start: the steady rate, lifted in group B while the paywall is new.
    trial_rate = np.where(is_b, PAID_RATE["B"], PAID_RATE["A"]) / TRIAL_TO_PAID
    novelty = 1.0 + NOVELTY_BOOST * np.exp(-day / NOVELTY_DECAY_DAYS)
    trial_rate = np.where(is_b, trial_rate * novelty, trial_rate)
    started_trial = rng.random(n) < trial_rate

    paid = started_trial & (rng.random(n) < TRIAL_TO_PAID)

    # Months paid: at least one, then a constant chance of cancelling before each month.
    churn = np.where(is_b, MONTHLY_CHURN["B"], MONTHLY_CHURN["A"])
    months = np.minimum(rng.geometric(churn), FOLLOW_UP_MONTHS)
    months_paid = np.where(paid, months, 0)

    price = np.where(is_b, PRICE["B"], PRICE["A"])
    revenue = np.round(price * months_paid, 2)

    return pd.DataFrame(
        {
            "user_id": [f"u_{i:06d}" for i in range(1, n + 1)],
            "group": group,
            "price_eur": price,
            "exposure_date": exposure_date.strftime("%Y-%m-%d"),
            "started_trial": started_trial.astype(int),
            "paid": paid.astype(int),
            "months_paid": months_paid,
            "revenue_12m": revenue,
        }
    )


def summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("group")
    out = pd.DataFrame(
        {
            "users": g.size(),
            "trial_rate_%": 100 * g["started_trial"].mean(),
            "paid_rate_%": 100 * g["paid"].mean(),
            "months_paid_per_payer": g["months_paid"].sum() / g["paid"].sum(),
            "revenue_12m_eur": g["revenue_12m"].sum(),
            "revenue_per_user_eur": g["revenue_12m"].mean(),
        }
    )
    return out.round(3)


if __name__ == "__main__":
    users = simulate()
    users.to_csv(OUT, index=False, lineterminator="\n")
    print(f"wrote {OUT.name}: {len(users):,} rows")
    print(summary(users).T.to_string())
