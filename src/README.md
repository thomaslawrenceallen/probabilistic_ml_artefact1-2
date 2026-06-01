# COMP3018 - Probabilistic Machine Learning
### Artefact 1 - US 10-Year Treasury Yield, Frequentist v Bayesian 

**Student :** Thomas Lawrence Allen  
**FAN :** alle0376  
**Date :** May 2026

 ---  
### Overview
This module implements the data ingestion, feature engineering, and 
probabilistic estimation pipeline for the US 10-Year Treasury Yield 
node in the group DAG-based probabilistic inference engine.

 --- 
### How to Run
1. Install Dependencies  
`pip install -r requirements.txt`

2. Run Pipeline  
`python artefact1.py`

 --- 
### Outputs
| File | Description |
|------|-------------|
| yield_eda.png | Three-panel EDA plot |
| yield_distribution.png | Distribution analysis and normality tests |
| freq_vs_bayes.png | Frequentist vs Bayesian comparison plot |
| us_10yr_yield.csv | Data contract output for team pipeline |

 --- 
### Limitations
- MCMC results subject to minor variation between runs
- yfinance may occasionally return empty data during market hours
- mismatched trading days between AUD and USD due to public holidays, data has been forward filled
- g++ compiler error warning on Windows is harmless, does not affect results, please ignore

 --- 
### Dependencies
See requirements.txt for full list. Key Packages;
- yfinance
- pymc
- scipy
- pandas
- matplotlib
