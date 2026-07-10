# AI-Powered Stock Market Intelligence & Trading Research Platform

An end-to-end machine learning research platform for stock market analysis,
direction prediction, signal generation, and strategy backtesting — built
from raw data collection through to an evaluated, tested trading strategy.

**Status:** Milestone 1 complete (data pipeline → features → targets →
model comparison → signal engine → backtest). Deep learning models (LSTM/GRU)
in progress.

## Disclaimer

This project is for educational and research purposes only. It does not
provide financial advice and does not guarantee investment returns. All
backtested results are historical and not indicative of future performance.

## Why this project

Most student ML projects stop at "I trained a model and got X% accuracy."
This project instead treats accuracy as only one part of the story — it
asks whether a model's predictions translate into a viable trading signal
after transaction costs, and it is built with the same rigor a real
quant research pipeline would need: no data leakage, time-aware
evaluation, and honest reporting of results (including two real bugs
found and fixed during development, documented below).

## Key results (AAPL, 2015–2025)

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Naive baseline | 53.7% | — |
| Logistic Regression | 53.9% | 0.588 |
| **Random Forest** | **59.1%** | **0.631** |
| XGBoost | 55.8% | 0.524 |

Backtest (Random Forest, unseen test period, July 2023–Dec 2024):
- Sharpe ratio: 1.04
- Max drawdown: -7.3%
- Strategy return: +6.7% vs. benchmark buy-and-hold: +36.4%

**Honest interpretation:** the model-driven strategy had solid risk-adjusted
returns but underperformed simple buy-and-hold in absolute terms during a
strong bull run, since it stayed in cash ~86% of the time. This is a
genuine, reportable finding about the tradeoff between risk management and
capturing upside — not a failure to hide.

## Bugs found and fixed during development

1. **Logistic Regression feature-scaling issue** — LR initially scored
   *worse than random* (ROC-AUC 0.37) due to unscaled features spanning
   wildly different ranges (e.g. OBV in the billions vs. RSI in 0-100).
   Fixed with `StandardScaler`, fit only on training data. ROC-AUC
   improved to 0.588.
2. **Overlapping-window backtest bug** — naively compounding daily rows
   of 5-day forward returns produced an impossible +358% benchmark
   return over 1.5 years. Root cause: rows overlap (each spans 5 days
   but occurs daily), so chaining them double-counts calendar days.
   Fixed by resampling to non-overlapping windows before computing
   cumulative metrics.

## Tech stack

- Python 3.12, Pandas, NumPy
- Scikit-learn, XGBoost
- PyTorch (LSTM/GRU — in progress)
- yfinance (data collection)
- Pytest (26+ tests)
- Jupyter (EDA)

## Architecture
Raw Data (yfinance)
→ Validation (schema, OHLC integrity, duplicates)
→ Cleaning (dtype enforcement, sorting)
→ Feature Engineering (35 technical indicators)
→ Target Engineering (return / direction / 3-class, leakage-tested)
→ Time-aware Train/Val/Test Split
→ Model Training (Naive, LR, RF, XGBoost)
→ Signal Engine (probability → BUY/HOLD/SELL)
→ Backtest Engine (transaction costs, Sharpe, drawdown, benchmark comparison)

## Project structure
src/
data/            - market data downloader
preprocessing/   - validation and cleaning
features/        - technical indicator engineering
targets/         - prediction target engineering
ml/              - model training and time-aware splitting
backtesting/     - signal generation and backtest engine
notebooks/         - exploratory data analysis
scripts/           - runnable pipeline scripts (one per phase)
tests/             - 26+ automated tests, including explicit
leakage-prevention tests
data/              - raw/interim/processed (gitignored, reproducible
from scripts)

## Engineering principles followed

- Raw data is never modified or silently overwritten
- Time-series data is never randomly shuffled for splitting/evaluation
- Scalers/transformers are fit only on training data
- Every feature and target is tested for look-ahead bias
- Model comparisons include realistic backtesting, not just classification
  accuracy
- No inflated claims — results are reported honestly, including
  limitations

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running the pipeline

```powershell
python scripts\download_stock.py       # Phase 2: download raw data
python scripts\validate_data.py        # Phase 3: validate
python scripts\clean_data.py           # Phase 3: clean
python scripts\build_features.py       # Phase 5: feature engineering
python scripts\build_targets.py        # Phase 6: target engineering
python scripts\train_baseline_models.py    # Milestone 1: model comparison
python scripts\run_milestone1_backtest.py  # Milestone 1: backtest
```

## Running tests

```powershell
pytest tests\ -v
```

## Roadmap

- [x] Data collection, validation, cleaning
- [x] EDA
- [x] Feature engineering (35 technical indicators)
- [x] Target engineering (return, direction, 3-class)
- [x] Baseline models (Naive, LR, RF, XGBoost) + backtest
- [ ] LSTM / GRU sequence models
- [ ] News sentiment (FinBERT)
- [ ] Ensemble model
- [ ] SHAP explainability
- [ ] Risk analysis & portfolio optimization
- [ ] FastAPI backend + React dashboard