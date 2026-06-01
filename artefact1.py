# Student Name: Thomas Lawrence Allen
# Student FAN: alle0376
# File: artefact1.py
# Date: 15-05-2026
# Description: Data ingestion, EDA, and probabilistic estimation of
#              US 10-Year Treasury yield changes. Frequentist MLE vs Bayesian MCMC.


# IMPORTS

import warnings
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import pymc as pm
import arviz as az

warnings.filterwarnings("ignore")


# CONFIGURATION

START_DATE  = "2018-01-01"   # 7 years — captures pre-COVID, COVID, hike cycle
END_DATE    = "2025-01-01"
OUTPUT_PATH = "us_10yr_yield.csv"


# DATA INGESTION

def fetch_yfinance(start: str, end: str) -> pd.DataFrame:
    """
    Fetch US 10-Year Treasury yield via yfinance ticker ^TNX.

    Args:
        start: Start date string (YYYY-MM-DD).
        end:   End date string (YYYY-MM-DD).

    Returns:
        DataFrame with daily yield (Close) indexed by Date.
    """

    raw = yf.download("^TNX", start=start, end=end, auto_adjust=True, progress=False)

    if raw.empty:
        print("WARNING: yfinance returned no data.")
        return pd.DataFrame()

    # Extract Close and flatten MultiIndex if present
    if isinstance(raw.columns, pd.MultiIndex):
        df = raw["Close"].rename(columns={"^TNX": "yield_pct"})
    else:
        df = raw[["Close"]].rename(columns={"Close": "yield_pct"})

    df.index.name = "Date"
    df.index = pd.to_datetime(df.index)
    print(f"yfinance: fetched {len(df)} trading days  |  "
          f"range: {df.index[0].date()} -> {df.index[-1].date()}")
    return df


# DATA VALIDATION


def validate_data(yield_df: pd.DataFrame) -> None:
    """
    Print a data quality report for the yield DataFrame.

    Args:
        yield_df: DataFrame with yield_pct column indexed by Date.
    """

    print("=== Data Quality Report ===")
    print(f"Shape:          {yield_df.shape}")
    print(f"Date range:     {yield_df.index.min().date()} -> {yield_df.index.max().date()}")
    print(f"Missing values: {yield_df['yield_pct'].isna().sum()}")
    print(f"Missing %:      {yield_df['yield_pct'].isna().mean()*100:.2f}%")
    print()
    print("=== Descriptive Statistics ===")
    print(yield_df["yield_pct"].describe().round(4))

    # Sanity check 
    out_of_range = yield_df[
        (yield_df["yield_pct"] < 0) | (yield_df["yield_pct"] > 15)
    ]
    if not out_of_range.empty:
        print(f"\nWARNING: {len(out_of_range)} out-of-range values detected:")
        print(out_of_range)
    else:
        print("\nRange check PASSED — all values within expected bounds (0–15%).")



# FEATURE ENGINEERING

def engineer_features(yield_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily changes and rolling volatility from yield level.

    First differencing removes the temporal trend, producing a stationary
    series suitable for distribution fitting. Rolling volatility captures
    time-varying uncertainty (volatility clustering).

    Args:
        yield_df: DataFrame with yield_pct column.

    Returns:
        DataFrame with yield_pct, daily_change, and vol_21d columns.
    """
    df = yield_df.copy()
    df["daily_change"] = df["yield_pct"].diff()
    df["vol_21d"]      = df["daily_change"].rolling(21).std()
    return df



# EDA — PLOT 1: YIELD LEVEL, DAILY CHANGES, ROLLING VOLATILITY


def plot_eda(yield_df: pd.DataFrame, save_path: str = "../data/raw/yield_eda.png") -> None:
    """
    Generate three-panel EDA plot: yield level, daily changes, rolling volatility.

    Args:
        yield_df:  DataFrame with yield_pct, daily_change, vol_21d columns.
        save_path: File path to save the plot.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(
        "US 10-Year Treasury Yield (^TNX) — EDA",
        fontsize=13, fontweight="bold"
    )

    # Panel 1: Yield level with regime shading
    ax = axes[0]
    ax.plot(yield_df.index, yield_df["yield_pct"], color="steelblue", lw=1.2)
    ax.axhspan(0,   1.5, alpha=0.08, color="green", label="Near-zero rate regime")
    ax.axhspan(3.5, 6,   alpha=0.08, color="red",   label="Hiking cycle regime")
    ax.set_title("Daily Yield Level (%)")
    ax.set_ylabel("Yield (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 2: Daily change in yield (first difference)
    ax = axes[1]
    ax.bar(
        yield_df.index,
        yield_df["daily_change"],
        color=["tomato" if x < 0 else "steelblue" for x in yield_df["daily_change"]],
        width=1,
        alpha=0.7,
    )
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Daily Change in Yield (bp equivalent)")
    ax.set_ylabel("Change (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(alpha=0.3)

    # Panel 3: 21-day rolling volatility
    ax = axes[2]
    ax.fill_between(yield_df.index, yield_df["vol_21d"], color="darkorange", alpha=0.6)
    ax.set_title("21-Day Rolling Volatility of Daily Yield Changes")
    ax.set_ylabel("Std Dev (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"EDA plot saved -> {save_path}")



# EDA — PLOT 2: DISTRIBUTION ANALYSIS


def plot_distribution(
    changes: pd.Series,
    mu_mle: float,
    sigma_mle: float,
    df_t: float,
    loc_t: float,
    scale_t: float,
    save_path: str = "../data/raw/yield_distribution.png",
) -> None:
    """
    Generate distribution analysis plot: histogram with MLE fits, Q-Q plot,
    and normality test summary.

    Args:
        changes:    Pandas Series of daily yield changes.
        mu_mle:     Normal MLE mean estimate.
        sigma_mle:  Normal MLE std estimate.
        df_t:       Student-t MLE degrees of freedom.
        loc_t:      Student-t MLE location.
        scale_t:    Student-t MLE scale.
        save_path:  File path to save the plot.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Distribution of Daily Yield Changes", fontsize=12, fontweight="bold")

    # Panel 1: Histogram with MLE fits
    ax = axes[0]
    x  = np.linspace(changes.min(), changes.max(), 300)
    ax.hist(changes, bins=60, density=True, alpha=0.5, color="steelblue", label="Empirical")
    ax.plot(x, stats.norm.pdf(x, mu_mle, sigma_mle),
            "r-", lw=2, label=f"Normal MLE\nμ={mu_mle:.4f}, σ={sigma_mle:.4f}")
    ax.plot(x, stats.t.pdf(x, df_t, loc_t, scale_t),
            "g--", lw=2, label=f"Student-t MLE\ndf={df_t:.1f}")
    ax.set_title("Histogram + MLE Fits")
    ax.set_xlabel("Daily Change (%)")
    ax.legend(fontsize=7)

    # Panel 2: Q-Q plot against Normal
    ax = axes[1]
    (osm, osr), (slope, intercept, r) = stats.probplot(changes, dist="norm")
    ax.scatter(osm, osr, s=4, alpha=0.4, color="steelblue")
    ax.plot(osm, slope * np.array(osm) + intercept,
            "r-", lw=1.5, label=f"R²={r**2:.4f}")
    ax.set_title("Normal Q-Q Plot")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 3: Normality test summary
    ax = axes[2]
    ax.axis("off")
    skew            = float(changes.skew())
    kurt            = float(changes.kurtosis())
    ks_stat, ks_p   = stats.kstest(changes, "norm", args=(mu_mle, sigma_mle))
    sw_stat, sw_p   = stats.shapiro(
        changes.sample(min(len(changes), 5000), random_state=42)
    )
    summary_text = (
        f"  n observations:    {len(changes)}\n"
        f"  Mean:              {changes.mean():.5f}%\n"
        f"  Std Dev:           {changes.std():.5f}%\n"
        f"  Skewness:          {skew:.4f}\n"
        f"  Excess Kurtosis:   {kurt:.4f}\n"
        f"  (Normal = 0; >0 = heavy tails)\n\n"
        f"  KS Test (Normal):\n"
        f"    statistic = {ks_stat:.4f}\n"
        f"    p-value   = {ks_p:.4f}\n"
        f"  {'REJECT normality' if ks_p < 0.05 else 'Cannot reject normality'} at α=0.05\n\n"
        f"  Shapiro-Wilk Test:\n"
        f"    statistic = {sw_stat:.4f}\n"
        f"    p-value   = {sw_p:.4f}\n"
        f"  {'REJECT normality' if sw_p < 0.05 else 'Cannot reject normality'} at α=0.05"
    )
    ax.text(
        0.05, 0.95, summary_text,
        transform=ax.transAxes, fontsize=9,
        verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )
    ax.set_title("Normality Tests")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Distribution plot saved -> {save_path}")

    # Key insight
    print("\n=== KEY INSIGHT FOR REPORT ===")
    if ks_p < 0.05 or sw_p < 0.05:
        print(f"Normality REJECTED, excess kurtosis = {kurt:.2f}")
        print("Daily yield changes have heavier tails than a Normal distribution.")
    else:
        print("Normality not rejected at α=0.05.")


# FREQUENTIST MLE

def fit_frequentist(changes: pd.Series) -> dict:
    """
    Fit Normal and Student-t distributions to daily yield changes via MLE.
    Compute 95% confidence intervals for Normal MLE parameters.

    Args:
        changes: Pandas Series of daily yield changes.

    Returns:
        Dict of MLE estimates and confidence intervals.
    """
    # Normal MLE
    mu_mle, sigma_mle = stats.norm.fit(changes)

    # Student-t MLE
    df_t, loc_t, scale_t = stats.t.fit(changes)

    # 95% confidence intervals for Normal MLE parameters
    n    = len(changes)
    z95  = stats.norm.ppf(0.975)
    se_mu    = sigma_mle / np.sqrt(n)
    se_sigma = sigma_mle / np.sqrt(2 * (n - 1))

    print(f"Normal MLE:    μ={mu_mle:.6f}, σ={sigma_mle:.6f}")
    print(f"Student-t MLE: df={df_t:.2f}, loc={loc_t:.6f}, scale={scale_t:.6f}")
    print(f"MLE mu:    {mu_mle:.6f}  95% CI: "
          f"[{mu_mle - z95*se_mu:.6f}, {mu_mle + z95*se_mu:.6f}]")
    print(f"MLE sigma: {sigma_mle:.6f}  95% CI: "
          f"[{sigma_mle - z95*se_sigma:.6f}, {sigma_mle + z95*se_sigma:.6f}]")

    return {
        "mu_mle":    mu_mle,
        "sigma_mle": sigma_mle,
        "df_t":      df_t,
        "loc_t":     loc_t,
        "scale_t":   scale_t,
        "z95":       z95,
        "se_mu":     se_mu,
        "se_sigma":  se_sigma,
        "n":         n,
    }



# BAYESIAN MCMC


def fit_bayesian(changes_np: np.ndarray, draws: int = 2000, tune: int = 1000) -> object:
    """
    Fit a Bayesian Student-t model to daily yield changes via MCMC (PyMC / NUTS).

    Prior specification (weakly informative):
      mu    ~ Normal(0, 0.05)    — yield changes centred near zero drift
      sigma ~ HalfNormal(0.05)   — positive only, typical daily vol ~0.06%
      nu    ~ Gamma(2, 0.1)      — degrees of freedom, allows model to learn tail weight

    Args:
        changes_np: Numpy array of daily yield changes (required by PyMC).
        draws:      Number of posterior draws per chain.
        tune:       Number of tuning steps per chain.

    Returns:
        ArviZ InferenceData trace object containing posterior samples.
    """
    with pm.Model():
        # Priors
        mu    = pm.Normal("mu",    mu=0,    sigma=0.05)
        sigma = pm.HalfNormal("sigma", sigma=0.05)
        nu    = pm.Gamma("nu",    alpha=2,  beta=0.1)

        # Student-t likelihood, appropriate given excess kurtosis > 0
        pm.StudentT("likelihood", mu=mu, sigma=sigma, nu=nu, observed=changes_np)

        # NUTS sampler — target_accept=0.95 
        trace = pm.sample(
            draws=draws,
            tune=tune,
            target_accept=0.95,
            return_inferencedata=True,
        )

    summary = az.summary(trace, var_names=["mu", "sigma", "nu"], round_to=4)

    # Convergence check
    print("\n=== Convergence Diagnostics ===")
    for param in ["mu", "sigma", "nu"]:
        r_hat  = float(summary.loc[param, "r_hat"])
        status = "✓ CONVERGED" if r_hat < 1.01 else "✗ CHECK CONVERGENCE"
        print(f"  R-hat {param}: {r_hat:.4f}  {status}")

    return trace



# COMPARISON PLOT — FREQUENTIST VS BAYESIAN


def plot_comparison(
    freq: dict,
    trace: object,
    save_path: str = "freq_vs_bayes.png",
) -> tuple:
    """
    Plot posterior histograms for μ, σ, ν with MLE point estimates overlaid.
    Print credible intervals and full comparison table.

    Args:
        freq:      Output dict from fit_frequentist().
        trace:     ArviZ InferenceData trace from fit_bayesian().
        save_path: File path to save the plot.

    Returns:
        Tuple of (mu_samples, sigma_samples, nu_samples) numpy arrays.
    """
    mu_samples    = trace.posterior["mu"].values.flatten()
    sigma_samples = trace.posterior["sigma"].values.flatten()
    nu_samples    = trace.posterior["nu"].values.flatten()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(
        "Frequentist MLE vs Bayesian Posterior — US 10-Year Yield Changes",
        fontsize=12, fontweight="bold",
    )

    # Panel 1: μ posterior
    axes[0].hist(mu_samples, bins=60, density=True, color="steelblue", alpha=0.7)
    axes[0].axvline(mu_samples.mean(), color="steelblue", lw=2,
                    label=f"Posterior mean = {mu_samples.mean():.6f}")
    axes[0].axvline(freq["mu_mle"], color="red", linestyle="--", lw=2,
                    label=f"MLE μ = {freq['mu_mle']:.6f}")
    axes[0].set_title("Posterior: μ (mean daily change)")
    axes[0].set_xlabel("μ")
    axes[0].legend(fontsize=8)

    # Panel 2: σ posterior
    axes[1].hist(sigma_samples, bins=60, density=True, color="darkorange", alpha=0.7)
    axes[1].axvline(sigma_samples.mean(), color="darkorange", lw=2,
                    label=f"Posterior mean = {sigma_samples.mean():.6f}")
    axes[1].axvline(freq["sigma_mle"], color="red", linestyle="--", lw=2,
                    label=f"MLE σ = {freq['sigma_mle']:.6f}")
    axes[1].set_title("Posterior: σ (volatility)")
    axes[1].set_xlabel("σ")
    axes[1].legend(fontsize=8)

    # Panel 3: ν posterior
    axes[2].hist(nu_samples, bins=60, density=True, color="mediumpurple", alpha=0.7)
    axes[2].axvline(nu_samples.mean(), color="mediumpurple", lw=2,
                    label=f"Posterior mean = {nu_samples.mean():.2f}")
    axes[2].axvline(freq["df_t"], color="red", linestyle="--", lw=2,
                    label=f"MLE df = {freq['df_t']:.2f}")
    axes[2].set_title("Posterior: ν (degrees of freedom)")
    axes[2].set_xlabel("ν")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison plot saved -> {save_path}")

    # Credible intervals
    print("\n=== Bayesian 94% Credible Intervals ===")
    for param, samples in [
        ("mu",    mu_samples),
        ("sigma", sigma_samples),
        ("nu",    nu_samples),
    ]:
        lower = np.percentile(samples, 3)
        upper = np.percentile(samples, 97)
        print(f"  {param}: [{lower:.6f}, {upper:.6f}]")

    # Full comparison table
    mu_mle    = freq["mu_mle"]
    sigma_mle = freq["sigma_mle"]
    df_t      = freq["df_t"]
    z95       = freq["z95"]
    se_mu     = freq["se_mu"]
    se_sigma  = freq["se_sigma"]

    print("\n" + "=" * 95)
    print(f"{'Parameter':<12} {'MLE Estimate':>14} {'MLE 95% CI':>22} "
          f"{'Bayes Mean':>12} {'Bayes 94% CI':>22}")
    print("=" * 95)
    print(
        f"{'mu':<12} {mu_mle:>14.6f} "
        f"[{mu_mle-z95*se_mu:.6f}, {mu_mle+z95*se_mu:.6f}] "
        f"{mu_samples.mean():>12.6f} "
        f"[{np.percentile(mu_samples,3):.6f}, {np.percentile(mu_samples,97):.6f}]"
    )
    print(
        f"{'sigma':<12} {sigma_mle:>14.6f} "
        f"[{sigma_mle-z95*se_sigma:.6f}, {sigma_mle+z95*se_sigma:.6f}] "
        f"{sigma_samples.mean():>12.6f} "
        f"[{np.percentile(sigma_samples,3):.6f}, {np.percentile(sigma_samples,97):.6f}]"
    )
    print(
        f"{'nu':<12} {df_t:>14.2f} "
        f"{'N/A':>22} "
        f"{nu_samples.mean():>12.2f} "
        f"[{np.percentile(nu_samples,3):.2f}, {np.percentile(nu_samples,97):.2f}]"
    )
    print("=" * 95)

    return mu_samples, sigma_samples, nu_samples


# EXPORT — DATA CONTRACT

def export_data_contract(yield_df: pd.DataFrame, output_path: str) -> None:
    """
    Export clean feature CSV for consumption by team pipeline modules.

    Output columns:
        yield_pct    — daily closing yield level (%)
        daily_change — first difference of yield (percentage points)
        vol_21d      — 21-day rolling std of daily changes (uncertainty proxy)

    Args:
        yield_df:    DataFrame with all engineered features.
        output_path: CSV file path.
    """
    output = yield_df[["yield_pct", "daily_change", "vol_21d"]].copy()
    output = output.ffill()   # Forward-fill US holidays to align with ASX calendar
    output.to_csv(output_path)
    print(f"Data contract saved -> {output_path}  shape={output.shape}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"Date range: {START_DATE} -> {END_DATE}")
    print("=" * 60)

    # 1. Ingest
    yield_df = fetch_yfinance(START_DATE, END_DATE)
    if yield_df.empty:
        print("ERROR: No data ingested. Exiting pipeline.")
        return
    
    # 2. Feature engineering
    yield_df = engineer_features(yield_df)

    # 3. Validation
    validate_data(yield_df)
    # 4. EDA plots
    plot_eda(yield_df, save_path="yield_eda.png")

    # 5. Frequentist MLE
    # Pandas Series — used for MLE fitting and plotting
    changes = yield_df["daily_change"].dropna()
    freq    = fit_frequentist(changes)

    # Distribution plot (uses MLE results)
    plot_distribution(
        changes,
        freq["mu_mle"], freq["sigma_mle"],
        freq["df_t"], freq["loc_t"], freq["scale_t"],
        save_path="yield_distribution.png",
    )

    # 6. Bayesian MCMC
    # Numpy array — required by PyMC
    changes_np = changes.values
    trace      = fit_bayesian(changes_np)

    # 7. Comparison plot + summary table
    plot_comparison(freq, trace, save_path="freq_vs_bayes.png")
    # 8. Export data contract
    export_data_contract(yield_df, OUTPUT_PATH)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()