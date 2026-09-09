# COMP3018 — Probabilistic Machine Learning

**Thomas Lawrence Allen** · FAN `alle0376` · Flinders University, Tonsley

Two artefacts modelling the **US 10-Year Treasury Yield** as a node in a group
DAG-based probabilistic inference engine, which estimates the conditional
probability of ASX energy stock price movements. Both artefacts here are the
individual contribution to that engine: the yield node and its regime
structure.

They are meant to be read in order — Artefact 2 consumes Artefact 1's output.

---

## The two artefacts

### [`A1_alle0376`](A1_alle0376) — Frequentist vs Bayesian estimation

Data ingestion, feature engineering, and probabilistic estimation for the yield
node. Estimates the same quantities two ways — classical point estimates
against a Bayesian posterior sampled with MCMC — and compares what each
approach tells you.

Produces `us_10yr_yield.csv`, the data contract the rest of the pipeline reads.

Built with `pymc`, `scipy`, `yfinance`, `pandas`, `matplotlib`.

### [`A2_alle0376`](A2_alle0376) — Hidden Markov Model for regime detection

Fits a Gaussian HMM over the yield series to recover latent macroeconomic
regimes, then tests whether the decoded regime carries any conditional
relationship with Woodside Energy (`WDS.AX`) opening prices on the ASX.

Baum-Welch for fitting, BIC for model selection (K=3 chosen), Viterbi and
Forward-Backward for decoding.

Two findings worth noting:

- **The transition matrix has structure.** Direct stable ↔ hiking/crisis
  transitions carry zero probability — every path between them routes through
  the transitional regime.
- **Regime explains volatility, not direction.** P(WDS opens higher | regime)
  sits at 0.54–0.55 across all three regimes, so the regime label has little
  next-day directional power despite fitting same-day volatility well.

---

## How they chain

```
A1: artefact1.py  ──►  Misc/us_10yr_yield.csv  ──►  A2: artefact2.py
                       (data contract)                      │
                                                            ▼
                                            Output/asx_energy_hmm.csv
                                            (enriched contract, feeds the DAG)
```

Artefact 2 will not run without `us_10yr_yield.csv` in place, so run Artefact 1
first if you're starting from scratch.

---

## Running either one

Each artefact has its own `README.md` and `requirements.txt` with full
instructions, outputs, and caveats — start there. The short version:

```bash
cd A1_alle0376          # or A2_alle0376
pip install -r requirements.txt
cd Source
python artefact1.py     # or artefact2.py
```

Python 3.10+. Both fetch market data through `yfinance` at runtime, so an
internet connection is required, and Yahoo Finance occasionally returns empty
responses for ASX tickers — retry after a few minutes if that happens.

MCMC results in Artefact 1 vary slightly between runs by nature.