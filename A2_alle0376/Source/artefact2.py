# Student Name: Thomas Lawrence Allen
# Student FAN: alle0376
# File: artefact2.py
# Date: 10/6/26
# Description: Hidden Markov Model for US 10-Year Treasury Yield regime detection
#              and conditional inference of ASX energy stock price movements.


# IMPORTS

import warnings
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from hmmlearn import hmm
import yfinance as yf

warnings.filterwarnings("ignore")


# CONFIGURATION

BASE_DIR        = pathlib.Path(__file__).parent.parent  # A2_ALLE0376/
MISC_DIR        = BASE_DIR / "Misc"
OUTPUT_DIR      = BASE_DIR / "Output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

YIELD_DATA_PATH = str(MISC_DIR / "us_10yr_yield.csv")
ASX_TICKER      = "WDS.AX"
ASX_START       = "2018-01-01"
ASX_END         = "2025-01-01"
OUTPUT_PATH     = str(OUTPUT_DIR / "asx_energy_hmm.csv")
N_STATES        = 3
N_ITER          = 1000



pathlib.Path("../output").mkdir(parents=True, exist_ok=True)


# DATA LOADING

def load_yield_data(path: str) -> pd.DataFrame:
    """
    Load the Artefact 1 data contract CSV.

    Args:
        path: File path to us_10yr_yield.csv.

    Returns:
        DataFrame with yield_pct, daily_change, vol_21d indexed by Date.
    """
    try:
        df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
        df = df.dropna(subset=['daily_change'])
        print(f"Yield data loaded: {len(df)} trading days | "
              f"{df.index.min().date()} -> {df.index.max().date()}")
        return df
    except Exception as e:
        print(f"Error loading yield data: {e}")
        return pd.DataFrame()



def fetch_asx_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch ASX energy stock prices via yfinance.

    Args:
        ticker: ASX ticker string e.g. 'WDS.AX'.
        start:  Start date string (YYYY-MM-DD).
        end:    End date string (YYYY-MM-DD).

    Returns:
        DataFrame with asx_close, asx_open, asx_return indexed by Date.
    """
    try:
        raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

        if raw.empty:
            print(f"WARNING: yfinance returned no data for {ticker}.")
            return pd.DataFrame()

        # Flatten MultiIndex if present, consistent with Artefact 1
        if isinstance(raw.columns, pd.MultiIndex):
            df = raw[["Close", "Open"]].copy()
            df.columns = ["asx_close", "asx_open"]
        else:
            df = raw[["Close", "Open"]].rename(
                columns={"Close": "asx_close", "Open": "asx_open"}
            )

        df.index.name = "Date"
        df.index = pd.to_datetime(df.index)

        # Daily log return — more statistically tractable than simple return
        df["asx_return"] = np.log(df["asx_close"] / df["asx_close"].shift(1))
        df = df.dropna(subset=["asx_return"])

        print(f"ASX data loaded ({ticker}): {len(df)} trading days | "
              f"{df.index.min().date()} -> {df.index.max().date()}")
        return df

    except Exception as e:
        print(f"Error fetching ASX data: {e}")
        return pd.DataFrame()


# DATA ALIGNMENT

def align_data(yield_df: pd.DataFrame, asx_df: pd.DataFrame) -> pd.DataFrame:
    """
    Align yield and ASX data on a common date index.

    US and ASX markets have different holiday calendars. Forward-fill
    is applied to handle gaps, consistent with Artefact 1 data contract.

    Args:
        yield_df: Yield DataFrame from load_yield_data().
        asx_df:   ASX DataFrame from fetch_asx_data().

    Returns:
        Merged DataFrame on inner join of trading dates, with all features.
    """
    # Inner join on dates both markets traded — drops days unique to either calendar
    df = yield_df.join(asx_df, how="inner")

    # Forward-fill any residual gaps from holiday misalignment
    df = df.ffill()

    # Drop any remaining NaNs (e.g. vol_21d at start of series)
    df = df.dropna()

    print(f"Aligned dataset: {len(df)} common trading days | "
          f"{df.index.min().date()} -> {df.index.max().date()}")
    print(f"Columns: {df.columns.tolist()}")
    return df 

# Feature Engineering - Next-Day Target

def add_next_day_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add next-day ASX open return as a target variable.

    The US bond market closes before the ASX opens, so today's
    overnight yield daily_change is known information at the time
    of tomorrow's ASX open. This shifts the open return back one
    day so it aligns with today's yield observation as a target.

    Args:
        df: Aligned DataFrame from align_data().

    Returns:
        DataFrame with additional columns: asx_open_return,
        target_open_return, target_direction.
    """
    df = df.copy()
    df["asx_open_return"]    = np.log(df["asx_open"] / df["asx_close"].shift(1))
    df["target_open_return"] = df["asx_open_return"].shift(-1)
    df["target_direction"]   = (df["target_open_return"] > 0).astype(float)
    df.loc[df["target_open_return"].isna(), "target_direction"] = np.nan

    n_no_target = df["target_open_return"].isna().sum()
    print(f"Next-day target constructed | {n_no_target} row(s) without a forward "
          f"target (expected: 1, the final trading day)")
    return df

# HMM — OBSERVATION MATRIX

def build_observations(df: pd.DataFrame) -> np.ndarray:
    """
    Construct the observation matrix for the HMM.

    Uses daily_change and vol_21d as the multivariate emission sequence,
    consistent with Artefact 1 feature engineering. Rows are timesteps,
    columns are features.

    Args:
        df: Aligned DataFrame from align_data().

    Returns:
        2D numpy array of shape (T, n_features).
    """
    obs = df[["daily_change", "vol_21d"]].values
    print(f"Observation matrix shape: {obs.shape}")
    print(obs[:10]) # Show first 10 rows for sanity check
    return obs


# HMM — MODEL FITTING

def fit_hmm(obs: np.ndarray, n_states: int, n_iter: int) -> hmm.GaussianHMM:
    vol = obs[:, 1]
    percentile_boundaries = np.linspace(0, 100, n_states + 1)
    masks = [
        (vol >= np.percentile(vol, percentile_boundaries[i])) &
        (vol <  np.percentile(vol, percentile_boundaries[i + 1]))
        for i in range(n_states)
    ]
    # Ensure last bin is inclusive
    masks[-1] = vol >= np.percentile(vol, percentile_boundaries[-2])

    means_init = np.array([
        obs[mask].mean(axis=0) if mask.sum() > 0 else obs.mean(axis=0)
        for mask in masks
    ])

    # Uniform startprob across however many states are requested
    startprob_init = np.full(n_states, 1.0 / n_states)
    startprob_init[0] = 0.5  # stable regime still weighted toward start of series
    startprob_init[1:] = (1 - 0.5) / (n_states - 1)

    # Diagonal-dominant transmat scaled to n_states
    off_diag = 0.10 / (n_states - 1)
    transmat_init = np.full((n_states, n_states), off_diag)
    np.fill_diagonal(transmat_init, 0.90)

    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=n_iter,
        init_params="c",
        params="stmc",
    )

    model.startprob_ = startprob_init
    model.transmat_  = transmat_init
    model.means_     = means_init

    model.fit(obs)
    return model


# HMM — CONVERGENCE CHECK

def check_convergence(model: hmm.GaussianHMM) -> None:
    """
    Print convergence diagnostics, consistent with Artefact 1 R-hat reporting.

    Args:
        model: Fitted GaussianHMM from fit_hmm().
    """
    print("\n=== HMM Convergence Diagnostics ===")
    if model.monitor_.converged:
        print(f" Status:      Converged after {model.monitor_.iter} iterations")
    else:
        print(f" Status:      Not converged after {model.monitor_.iter} iterations")    
    print(f" Log Likelihood: {model.monitor_.history[-1]:.4f}")


# HMM — DECODING

def decode_regimes(model: hmm.GaussianHMM, obs: np.ndarray, df: pd.DataFrame) -> pd.DataFrame:
    """
    Run Viterbi decoding and Forward-Backward to attach regime labels
    and state probabilities to the aligned DataFrame.

    Args:
        model: Fitted GaussianHMM.
        obs:   Observation matrix.
        df:    Aligned DataFrame to attach results to.

    Returns:
        DataFrame with additional columns: regime, prob_state_0..K,
        and regime_label (string name).
    """
    # Viterbi - most probable hidden state sequence
    _, state_sequence = model.decode(obs, algorithm="viterbi")

    # Forward-Backward - marginal probabilities of each state at each time
    state_probs = model.predict_proba(obs)

    df = df.copy()
    df["regime"] = state_sequence

    n_states = model.n_components
    for i in range(n_states):
        df[f"prob_state_{i}"] = state_probs[:, i]

    # label states nased on ascending volatility of their emission distribution
    # so labels are deterministic regardless of arbitrary hmmlearn state ordering
    state_vols = model.covars_[:,1,1] # variance of vold_21d feature per state
    order = np.argsort(state_vols) # 0=lowest vol, 2=highest vol
    label_map = {order[0]: "stable", order[1]: "transitional", order[2]: "hiking/crisis"}
    df["regime_label"] = df["regime"].map(label_map)

    print("=== Regime Distribution ===")
    print(df["regime_label"].value_counts())
    print()
    return df

def print_emmission_params(model: hmm.GaussianHMM) -> None:
    """
    Print the mean and covariance parameters of the fitted HMM for each state.

    Args:
        model: Fitted GaussianHMM.
    """
    state_vols = model.covars_[:, 1, 1]
    order = np.argsort(state_vols)
    label_map_temp = {order[0]: "stable", order[1]: "transitional", order[2]: "hiking/crisis"}

    print("=== Emission Parameters per State ===")
    for i in range(model.n_components):
        print(f"  State {i} ({label_map_temp[i]}):")
        print(f"    Mean daily_change: {model.means_[i][0]:.6f}")
        print(f"    Mean vol_21d:      {model.means_[i][1]:.6f}")
        print(f"    vol_21d variance:  {model.covars_[i][1][1]:.6f}")
    print()

def print_transition_matrix(model: hmm.GaussianHMM, label_map: dict) -> None:
    """
    Print the learned transition matrix with regime labels.

    Args:
        model:     Fitted GaussianHMM from fit_hmm().
        label_map: State index to regime label mapping from decode_regimes().
    """
    print("=== Transition Matrix ===")
    n = model.n_components
    labels = [label_map.get(i, f"state_{i}") for i in range(n)]

    # Header row
    header = f"{'':20}" + "".join(f"{l:20}" for l in labels)
    print(header)

    # Each row
    for i in range(n):
        row = f"{labels[i]:20}" + "".join(f"{model.transmat_[i][j]:.6f}{'':12}" for j in range(n))
        print(row)

    # Expected durations
    print("\n  Expected regime durations (trading days):")
    for i in range(n):
        duration = 1 / (1 - model.transmat_[i][i])
        print(f"    {labels[i]:20}: {duration:.1f} days")
    print()


# MODEL SELECTION — BIC

def compute_bic(model: hmm.GaussianHMM, obs: np.ndarray, n_params: int) -> float:
    """
    Compute Bayesian Information Criterion for model selection between K=2 and K=3.

    BIC = -2 * log_likelihood + n_params * log(T)

    Args:
        model:    Fitted GaussianHMM.
        obs:      Observation matrix.
        n_params: Number of free parameters for this model.

    Returns:
        BIC score (lower is better).
    """
    T = len(obs)
    log_likelihood = model.score(obs) 
    bic = -2 * log_likelihood + n_params * np.log(T)
    print(f"BIC: {bic:.2f} | Log Likelihood: {log_likelihood:.2f} | "
          f"n_params: {n_params} | T: {T}")
    return bic


# ASX CONDITIONAL ANALYSIS

def analyse_asx_by_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute conditional ASX return statistics per yield regime.

    For each hidden state, compute mean return, volatility, and
    P(positive return) of the ASX energy stock. This is the core
    output linking yield regimes to ASX energy stock movements.

    Args:
        df: Decoded DataFrame from decode_regimes().

    Returns:
        Summary DataFrame indexed by regime with conditional statistics.
    """
    rows = []
    for label, group in df.groupby("regime_label"):
        rows.append({
            "regime":            label,
            "n_days":            len(group),
            "mean_asx_return":   group["asx_return"].mean(),
            "std_asx_return":    group["asx_return"].std(),
            "p_positive_return": (group["asx_return"] > 0).mean(),
            "mean_yield_change": group["daily_change"].mean(),
            "mean_vol_21d":      group["vol_21d"].mean(),
        })

    summary = pd.DataFrame(rows).set_index("regime")
    print("=== Conditional ASX Return Statistics by Yield Regime ===")
    print(summary.round(6))
    print()
    return summary

# Next-Day Conditional Prediction
# Next-Day Conditional Prediction

def predict_next_open(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute P(WDS opens higher | yield regime) from the empirical
    conditional distribution of next-day open returns.

    Args:
        df: Decoded DataFrame from decode_regimes(), after
            add_next_day_target() has been applied.

    Returns:
        Summary DataFrame indexed by regime_label.
    """
    import scipy.stats as stats
    valid = df.dropna(subset=["target_open_return", "regime_label"])

    rows = []
    for label, group in valid.groupby("regime_label"):
        returns = group["target_open_return"]
        n = len(returns)
        ci_low, ci_high = stats.t.interval(0.95, n - 1, loc=returns.mean(), scale=stats.sem(returns))
        rows.append({
            "regime_label":       label,
            "n_obs":              n,
            "p_open_higher":      (returns > 0).mean(),
            "mean_target_return": returns.mean(),
            "std_target_return":  returns.std(),
            "ci_95_low":          ci_low,
            "ci_95_high":         ci_high,
        })

    summary = pd.DataFrame(rows).set_index("regime_label")
    print("=== P(Next-Day WDS Open Higher | Yield Regime) ===")
    print(summary.round(6))
    print()
    return summary


def predict_live(df: pd.DataFrame, prediction_table: pd.DataFrame) -> dict:
    """
    Predict next-day ASX open return based on the most recent yield regime.

    Args:
        df:               Decoded DataFrame from decode_regimes().
        prediction_table: Summary DataFrame from predict_next_open().

    Returns:
        Dict with predicted regime, probability, and expected return.
    """
    latest = df.iloc[-1]
    regime = latest["regime_label"]
    row    = prediction_table.loc[regime]

    result = {
        "as_of_date":           df.index[-1].date(),
        "current_regime":       regime,
        "p_open_higher":        row["p_open_higher"],
        "expected_mean_return": row["mean_target_return"],
        "ci_95":                (row["ci_95_low"], row["ci_95_high"]),
    }

    print("=== Next Trading Day Prediction ===")
    print(f"  As of:               {result['as_of_date']}")
    print(f"  Current regime:      {result['current_regime']}")
    print(f"  P(WDS opens higher): {result['p_open_higher']:.4f}")
    print(f"  Expected return:     {result['expected_mean_return']:.6f}")
    print(f"  95% CI:              [{result['ci_95'][0]:.6f}, {result['ci_95'][1]:.6f}]")
    return result

# PLOTTING — REGIME OVERLAY

def plot_regimes(df: pd.DataFrame, save_path: str = None) -> None:
    """
    Plot yield level with regime colour overlay and ASX returns by regime.

    Args:
        df:        Decoded DataFrame.
        save_path: Output file path.
    """
    if save_path is None:
        save_path = str(OUTPUT_DIR / "hmm_regimes.png")
    
    palette = {"stable": "darkmagenta", "transitional": "moccasin", "hiking/crisis": "red"}

    fig, axes = plt.subplots(3, 1, figsize=(14, 11))
    fig.suptitle(
        "US 10-Year Treasury Yield — HMM Regime Detection",
        fontsize=13, fontweight="bold"
    )

    # Panel 1: Yield level with regime shading
    ax = axes[0]
    ax.plot(df.index, df["yield_pct"], color="black", lw=1.0, zorder=2)
    for label, colour in palette.items():
        mask = df["regime_label"] == label
        ax.fill_between(df.index, df["yield_pct"].min(), df["yield_pct"].max(),
                        where=mask, alpha=0.25, color=colour, label=label)
    ax.set_title("Yield Level (%) with Decoded Regime Overlay")
    ax.set_ylabel("Yield (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 2: Forward-Backward state probabilities
    ax = axes[1]
    prob_cols = [c for c in df.columns if c.startswith("prob_state_")]
    colours   = ["darkmagenta", "moccasin", "red"]
    for col, colour in zip(prob_cols, colours):
        ax.plot(df.index, df[col], lw=0.8, alpha=0.8, color=colour, label=col)
    ax.set_title("Forward-Backward Marginal State Probabilities")
    ax.set_ylabel("P(state | observations)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 3: ASX daily log return coloured by regime
    ax = axes[2]
    for label, colour in palette.items():
        mask = df["regime_label"] == label
        ax.bar(df.index[mask], df["asx_return"][mask],
               color=colour, alpha=0.7, width=1, label=label)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(f"ASX Energy Daily Log Return by Yield Regime ({ASX_TICKER})")
    ax.set_ylabel("Log Return")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Regime plot saved -> {save_path}")


# PLOTTING — CONDITIONAL ASX DISTRIBUTIONS

def plot_asx_conditional(df: pd.DataFrame, save_path: str = None) -> None:
    """
    Plot distribution of ASX daily returns conditioned on each yield regime.
    Visualises P(ASX return | yield regime) for each hidden state.

    Args:
        df:        Decoded DataFrame.
        save_path: Output file path.
    """
    if save_path is None:
        save_path = str(OUTPUT_DIR / "asx_conditional.png")
    
    import scipy.stats as stats

    palette = {"stable": "steelblue", "transitional": "darkorange", "hiking/crisis": "tomato"}
    labels  = list(palette.keys())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(
        f"P(ASX Return | Yield Regime) — {ASX_TICKER}",
        fontsize=12, fontweight="bold"
    )

    for ax, label in zip(axes, labels):
        colour = palette[label]
        subset = df[df["regime_label"] == label]["asx_return"].dropna()

        if len(subset) < 10:
            ax.set_title(f"{label}\n(insufficient data)")
            continue

        ax.hist(subset, bins=40, density=True, alpha=0.5, color=colour)

        # Fit and overlay a Normal for reference
        mu, sigma = subset.mean(), subset.std()
        x = np.linspace(subset.min(), subset.max(), 200)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), color=colour, lw=2,
                label=f"μ={mu:.4f}\nσ={sigma:.4f}\nn={len(subset)}")

        ax.axvline(0, color="black", lw=0.8, linestyle="--")
        ax.set_title(f"Regime: {label}")
        ax.set_xlabel("Log Return")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Conditional distribution plot saved -> {save_path}")



# EXPORT — DATA CONTRACT

def export_data_contract(df: pd.DataFrame, output_path: str) -> None:
    """
    Export enriched CSV for downstream DAG modules (VIX, AUD/USD nodes).

    Output columns include all Artefact 1 features plus:
        regime         — Viterbi decoded hidden state (0, 1, 2)
        regime_label   — human-readable label (stable / crisis / hiking)
        prob_state_0/1/2 — forward-backward marginal probabilities

    Args:
        df:          Decoded DataFrame.
        output_path: CSV file path.
    """
    export_cols = [
        "yield_pct", "daily_change", "vol_21d",
        "asx_close", "asx_open", "asx_return",
        "target_open_return", "target_direction",
        "regime", "regime_label",
    ] + [c for c in df.columns if c.startswith("prob_state_")]

    output = df[export_cols].copy()
    output.to_csv(output_path)
    print(f"Data contract saved -> {output_path}  shape={output.shape}")



# MAIN

def main():
    yield_df = load_yield_data(YIELD_DATA_PATH)
    asx_df   = fetch_asx_data(ASX_TICKER, ASX_START, ASX_END)
    df       = align_data(yield_df, asx_df)
    df       = add_next_day_target(df)          # <- fixed name
    obs      = build_observations(df)

    print("\n=== BIC Model Selection ===")
    model_k2 = fit_hmm(obs, n_states=2, n_iter=N_ITER)
    bic_k2   = compute_bic(model_k2, obs, n_params=2*1 + 2*2 + 2*4)
    model_k3 = fit_hmm(obs, n_states=3, n_iter=N_ITER)
    bic_k3   = compute_bic(model_k3, obs, n_params=3*2 + 3*2 + 3*4)
    print(f"  BIC K=2: {bic_k2:.4f}")
    print(f"  BIC K=3: {bic_k3:.4f}")
    print(f"  Selected: K={'2' if bic_k2 < bic_k3 else '3'} (lower BIC wins)\n")

    model = model_k3
    check_convergence(model)

    df = decode_regimes(model, obs, df)          # regime_label created here
    print_emmission_params(model)               # print emission params
    print_transition_matrix(model, label_map={0: "stable", 1: "transitional", 2: "hiking/crisis"}) ## print transition matrix with labels
    summary = analyse_asx_by_regime(df)

    prediction_table = predict_next_open(df)       # <- fixed name, now runs after decode
    predict_live(df, prediction_table)

    plot_regimes(df)
    plot_asx_conditional(df)
    export_data_contract(df, OUTPUT_PATH)

    print("\nPipeline complete.")

if __name__ == "__main__":
    main()
        
