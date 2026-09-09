# Student Name: Thomas Lawrence Allen
# Student FAN: alle0376
# File: test_cases.py
# Date: [DATE]
# Description: Stress-test the fitted HMM at specific historical dates
#              using the exported decoded data contract from artefact2.py.

import pathlib
import pandas as pd
import numpy as np

BASE_DIR  = pathlib.Path(__file__).parent.parent
DATA_PATH = str(BASE_DIR / "Output" / "asx_energy_hmm.csv")


TEST_CASE_DATES = [
    "2020-03-16",   # COVID crash
    "2019-06-14",   # calm pre-COVID baseline
    "2023-10-19",   # peak of hiking cycle
    "2022-03-01",   # plausible ambiguous transition point
    "2024-07-11",   # most uncertain day

]


def load_decoded_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    print(f"Loaded decoded dataset: {len(df)} rows | "
          f"{df.index.min().date()} -> {df.index.max().date()}")
    return df


def run_test_cases(df: pd.DataFrame, test_dates: list) -> pd.DataFrame:
    prob_cols = [c for c in df.columns if c.startswith("prob_state_")]
    rows = []

    for date_str in test_dates:
        ts = pd.Timestamp(date_str)
        if ts not in df.index:
            print(f"WARNING: {date_str} not in dataset (holiday/weekend?) — skipping.")
            continue


        row = df.loc[ts]
        log_return = row.get("target_open_return", float("nan"))
        pct_return = (np.exp(log_return) - 1) * 100 if pd.notna(log_return) else float("nan")

        rows.append({
            "date":           ts.date(),
            "daily_change":   row["daily_change"],
            "vol_21d":        row["vol_21d"],
            "decoded_regime": row["regime_label"],
            **{c: row[c] for c in prob_cols},
            "max_uncertainty": 1 - row[prob_cols].max(),
            "next_day_open_return_actual": pct_return,
        })

    summary = pd.DataFrame(rows).set_index("date")
    print("=== Test Case Stress Results ===")
    print(summary.round(6))
    return summary


def find_most_uncertain_days(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Identify the n days where the model was least confident in its
    regime assignment — useful for finding a genuinely ambiguous
    test case rather than guessing a date blindly.
    """
    prob_cols = [c for c in df.columns if c.startswith("prob_state_")]
    uncertainty = 1 - df[prob_cols].max(axis=1)
    most_uncertain = uncertainty.sort_values(ascending=False).head(n)
    print(f"=== Top {n} Most Uncertain Days ===")
    print(most_uncertain.round(6))
    return most_uncertain


def main():
    df = load_decoded_data(DATA_PATH)
    find_most_uncertain_days(df)
    run_test_cases(df, TEST_CASE_DATES)


if __name__ == "__main__":
    main()