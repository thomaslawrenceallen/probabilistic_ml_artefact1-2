# COMP3018 – Probabilistic Machine Learning
## Artefact 2 — Hidden Markov Model for US 10-Year Treasury Yield Regime Detection

**Student:** Thomas Lawrence Allen  
**FAN:** alle0376  
**Unit:** COMP3018 Probabilistic Machine Learning  
**Institution:** Flinders University, Tonsley Campus  

---

## Project Overview

This project builds a Hidden Markov Model (HMM) to detect latent macroeconomic regimes in the US 10-Year Treasury Yield, and tests whether the detected regime carries any conditional relationship with Woodside Energy (WDS.AX) stock opening prices on the ASX.

This study forms part of a hypothetical collaborative probabilistic inference engine designed to estimate the conditional probability of ASX energy stock price movements using a Directed Acyclic Graph (DAG) of causal feature nodes. The yield regime detection module is the individual contribution.

---

## Directory Structure

```
A2_ALLE0376/
├── Misc/
│   └── us_10yr_yield.csv          # Artefact 1 data contract (required input)
├── Output/
│   ├── asx_energy_hmm.csv         # Enriched data contract (generated)
│   ├── hmm_regimes.png            # Regime overlay plot (generated)
│   └── asx_conditional.png        # Conditional return distributions (generated)
├── Reports/
│   ├── A2_alle0376_SelfAssessment.pdf 
│   └── A2_alle0376_Report.pdf    # Technical report
├── Source/
│   ├── artefact2.py               # Main HMM pipeline
│   └── test_cases.py              # Stress-test analysis on specific dates
└── README.md                      # This file
```

---

## Requirements

See `requirements.txt` for the full dependency list. Install all dependencies with:

```bash
pip install -r requirements.txt
```

**Python version:** 3.10 or later recommended. The project was developed on Python 3.14.

---

## How to Run

### Step 1 — Ensure the Artefact 1 data contract is present

The file `Misc/us_10yr_yield.csv` must exist before running. This is the data contract exported by Artefact 1 (`artefact1.py`). If it is not present, run Artefact 1 first to generate it.

### Step 2 — Run the main pipeline

From the `Source/` directory:

```bash
python artefact2.py
```

Or from the project root:

```bash
python Source/artefact2.py
```

This will:
1. Load the Artefact 1 yield data contract
2. Fetch Woodside Energy (WDS.AX) price data via yfinance
3. Align both datasets on common trading dates
4. Construct the next-day ASX open return target
5. Fit Gaussian HMMs for K=2 and K=3 via Baum-Welch
6. Select K=3 via BIC model comparison
7. Decode regimes via Viterbi and Forward-Backward
8. Compute conditional ASX return statistics per regime
9. Generate the next-day open prediction
10. Save plots and enriched data contract to `Output/`

**Expected runtime:** under 60 seconds on a standard laptop.

### Step 3 — Run the test cases

Ensure Step 2 has been completed and `Output/asx_energy_hmm.csv` exists, then:

```bash
python test_cases.py
```

This reads the exported data contract and stress-tests the model at five specific economically meaningful dates, reporting decoded regimes, state probabilities, uncertainty scores, and realised next-day returns.

---

## Output Files

| File | Description |
|------|-------------|
| `Output/asx_energy_hmm.csv` | Enriched data contract with regime labels, state probabilities, and next-day return targets for downstream DAG modules |
| `Output/hmm_regimes.png` | Three-panel plot: yield level with regime overlay, Forward-Backward state probabilities, and ASX returns coloured by regime |
| `Output/asx_conditional.png` | Conditional distribution of WDS.AX daily returns per yield regime |

---

## Key Results

- **K=3 selected** via BIC (score: −28,316,834 vs −26,213,499 for K=2)
- **Converged** in 14 Baum-Welch iterations
- **Regime distribution:** Stable 602 days, Transitional 586 days, Hiking/Crisis 517 days
- **Transition matrix:** Direct stable <-> hiking/crisis transitions have 0% probability — all transitions route through the transitional regime
- **P(WDS opens higher | regime):** 0.54–0.55 across all regimes, regime explains same-day volatility well but has limited next-day directional power

---

## Notes

- An internet connection is required to fetch WDS.AX data via yfinance
- If yfinance returns no data for WDS.AX, wait a few minutes and retry, Yahoo Finance occasionally has transient availability issues with ASX tickers
- All output paths are resolved relative to the script location using `pathlib.Path(__file__)`, so the script can be run from any working directory