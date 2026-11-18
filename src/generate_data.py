"""Generate the fixed synthetic microloan sample used in this repository.

All records are simulated. No real borrower or lender records are used.
"""
import numpy as np
import pandas as pd

MASTER_SEED = 20_260_101
SAMPLE_SEED = 4_250_779_039
MASTER_N = 3_000
SAMPLE_N = 900
N_DEFAULTS = 180


def make_master_dataset(n=MASTER_N, seed=MASTER_SEED, target_rate=0.20):
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(38, 10, n), 21, 65).round()
    income = np.clip(rng.lognormal(np.log(15_000), 0.45, n), 4_000, 90_000).round(-2)
    loan_amount = np.clip(income * rng.uniform(1.0, 4.0, n), 5_000, 150_000).round(-3)
    term_months = rng.choice([6, 12, 18, 24], size=n, p=[0.2, 0.4, 0.2, 0.2])
    prior_loans = np.clip(rng.poisson(1.5, n), 0, 8)
    agri_share = rng.beta(2, 2.5, n).round(2)
    distance_km = np.clip(rng.gamma(2.0, 4.0, n), 0.5, 40).round(1)
    savings_group = rng.binomial(1, 0.5, n)

    loan_to_income = loan_amount / income
    risk = (1.8 * (loan_to_income - 2.5)
            + 3.0 * (agri_share - 0.4)
            + 0.12 * (distance_km - 8)
            - 2.0 * savings_group
            - 0.60 * (prior_loans - 1.5)
            + 0.50 * (term_months - 12) / 6
            - 0.03 * (age - 38))

    low, high = -10.0, 10.0
    for _ in range(60):
        mid = (low + high) / 2
        if (1 / (1 + np.exp(-(mid + risk)))).mean() > target_rate:
            high = mid
        else:
            low = mid
    p_default = 1 / (1 + np.exp(-(low + risk)))
    default = (rng.random(n) < p_default).astype(int)

    df = pd.DataFrame({
        "age": age, "monthly_income": income, "loan_amount": loan_amount,
        "term_months": term_months, "prior_loans": prior_loans,
        "agri_income_share": agri_share, "branch_distance_km": distance_km,
        "savings_group": savings_group, "default": default,
    })
    df.loc[rng.random(n) < 0.03, "monthly_income"] = np.nan
    df.loc[rng.random(n) < 0.02, "branch_distance_km"] = np.nan
    return df


def make_sample(master=None, seed=SAMPLE_SEED):
    master = make_master_dataset() if master is None else master
    rng = np.random.default_rng(seed)
    positive = master.index[master["default"] == 1].to_numpy()
    negative = master.index[master["default"] == 0].to_numpy()
    chosen = np.concatenate([
        rng.choice(positive, N_DEFAULTS, replace=False),
        rng.choice(negative, SAMPLE_N - N_DEFAULTS, replace=False),
    ])
    rng.shuffle(chosen)
    return master.loc[chosen]


if __name__ == "__main__":
    sample = make_sample()
    out = "data/synthetic_microloan_sample.csv"
    sample.to_csv(out, index=True, index_label="master_row_id")
    print(f"Wrote {out}: {len(sample):,} rows; {int(sample.default.sum())} defaults")
