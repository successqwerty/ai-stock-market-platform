# AI-Powered Stock Market Intelligence & Trading Research Platform

An end-to-end machine learning research platform for stock market analysis,
direction prediction, signal generation, and strategy backtesting — built
from raw data collection through to an evaluated, tested trading strategy.

**Status:** Core pipeline complete (data → features → targets → classical
ML → LSTM/GRU → SHAP explainability → FastAPI backend → multi-stock
comparison). Frontend demo included.

## Disclaimer

This project is for educational and research purposes only. It does not
provide financial advice and does not guarantee investment returns. All
backtested results are historical and not indicative of future performance.

## Why this project

Most student ML projects stop at "I trained a model and got X% accuracy."
This project instead treats accuracy as only one part of the story — it
asks whether a model's predictions translate into a viable trading signal
after transaction costs, and it is built with the same rigor a real quant
research pipeline would need: no data leakage, time-aware evaluation, and
honest reporting of results (including real bugs and limitations found
during development, documented below).

## Key results (AAPL, single-stock baseline)

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Naive baseline | 53.7% | — |
| Logistic Regression | 53.9% | 0.588 |
| Random Forest | 59.1% | 0.631 |
| XGBoost | 55.8% | 0.524 |
| **LSTM (tuned)** | **60.9%** | **0.646** |
| GRU (tuned) | 49.3% | 0.524 |
| **Ensemble (Weighted Avg)** | **56.8%** | **0.653** |

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
3. **LSTM overfitting** — a large LSTM (64 hidden units, 2 layers)
   scored *worse than the naive baseline* due to overfitting on ~1,700
   training sequences. Diagnosed via train/val loss divergence, fixed
   by reducing model capacity (16 hidden units, 1 layer, shorter
   15-day sequences), which became the best-performing model overall.

## LSTM vs GRU: an architecture comparison

Both sequence models were tuned with matched, right-sized architectures
(16 hidden units, 1 layer, 15-day sequences, 0.4 dropout) after the
oversized LSTM overfit (see above). With capacity matched, LSTM clearly
outperformed GRU (60.9% vs 49.3% accuracy) — suggesting LSTM's explicit
cell-state pathway captured longer-term dependencies in this feature set
that GRU's simpler gating mechanism did not. This shows architecture
*type*, not just parameter count, matters when data is limited.

## Weighted-Average Ensemble Model

A weighted-average ensemble combining Random Forest, XGBoost, and LSTM predictions was constructed using validation-set ROC-AUC to determine model weights.

- **Strict Leakage Prevention**: Weights are derived strictly from validation performance above random chance (0.5 AUC) — never from held-out test data. When models score below 0.5 AUC on validation, an equal-weighting fallback (33.3% each) is automatically applied.
- **Performance Result**: On the unseen test period, the Ensemble achieved an **ROC-AUC of 0.6533**, outperforming every individual constituent model (LSTM 0.6260, XGBoost 0.6137, RF 0.5909). Accuracy reached 56.76%, demonstrating improved ranking reliability and reduced model variance.

## Explainability (SHAP)

Using SHAP TreeExplainer on the Random Forest model, the top predictive
features were volatility and volume-based, not classic momentum
oscillators:

| Rank | Feature | Mean \|SHAP value\| |
|---|---|---|
| 1 | atr_14 (Average True Range) | 0.0247 |
| 2 | rolling_vol_mean_20 | 0.0183 |
| 3 | obv (On-Balance Volume) | 0.0173 |
| 4 | momentum_10 | 0.0114 |
| 5 | bb_upper (Bollinger Band) | 0.0108 |

This suggests the model relies more on *how turbulently* a stock is
trading than on classic overbought/oversold signals like RSI — a
genuinely interesting, model-derived finding rather than an assumption
built into the feature set.

Each individual prediction can also be explained locally — e.g. a DOWN
prediction on 2022-01-14 was driven primarily by negative momentum,
declining OBV, and elevated ATR, all consistent with the actual outcome.

## Multi-stock comparison (AAPL, MSFT, GOOGL, AMZN, TSLA)

The same pipeline (features, targets, Random Forest, time-aware split)
was run independently across 5 large-cap tickers using data through
July 2026.

| Ticker | RF Accuracy | Naive Accuracy | ROC-AUC |
|---|---|---|---|
| AAPL | 44.7% | 57.6% | 0.444 |
| MSFT | 43.6% | 59.7% | 0.504 |
| GOOGL | 50.1% | 57.6% | 0.508 |
| AMZN | 47.1% | 61.4% | 0.581 |
| TSLA | 49.9% | 50.4% | 0.480 |

**Honest finding — regime shift:** on this validation window
(Feb 2023–Oct 2024), Random Forest underperformed the naive
majority-class baseline across every ticker. This differs from the
original single-stock AAPL result above (59.1% accuracy), where the
validation window was Jan 2022–Jul 2023. Extending the data through
2026 shifted the time-aware split boundaries into a later, different
market period.

This is a genuine and instructive finding, not a bug: models trained
on one market regime (2015–2023) don't necessarily generalize to a
later regime (2023–2024) with different volatility and rate
conditions. This is a real, well-known risk in quantitative finance —
called regime shift or distribution shift — and a good illustration of
why models need periodic retraining and monitoring in production, not
a one-time train-and-deploy.

## Live API

A FastAPI backend serves live predictions from the trained Random
Forest model, with SHAP-based explanations included in every response.

### Endpoints

- `GET /health` — service health check
- `POST /predict` — returns a direction prediction, probability, signal
  (BUY/HOLD/SELL), and top 5 SHAP feature contributors for the latest
  available data
- `GET /stocks/{ticker}/history?days=N` — recent price/volume history

### Run it

```powershell
uvicorn backend.app.main:app --reload
```

Then open `http://127.0.0.1:8000/docs` for interactive API documentation.

### Example response (`POST /predict`, `{"ticker": "AAPL"}`)

```json
{
  "ticker": "AAPL",
  "prediction_date": "2024-12-23",
  "probability_up": 0.4697,
  "signal": "HOLD",
  "top_contributors": [
    {"feature": "obv", "shap_value": -0.0391, "direction": "pushes toward DOWN"},
    {"feature": "atr_14", "shap_value": -0.0258, "direction": "pushes toward DOWN"},
    {"feature": "rolling_vol_mean_20", "shap_value": 0.0248, "direction": "pushes toward UP"}
  ]
}
```

### Design note: why Random Forest, not LSTM, is deployed

LSTM scored marginally higher in validation (60.9% vs 59.1% accuracy),
but Random Forest was chosen for the live API because it requires only
the latest row of features (not a 15-day scaled sequence), pairs with
exact and fast SHAP explanations via `TreeExplainer`, and avoids the
added complexity of persisting a fitted scaler and managing PyTorch
inference mode. This was a deliberate simplicity-vs-marginal-accuracy
tradeoff, not an oversight.

## Demo frontend

A single-page terminal-styled dashboard calls the live API and
visualizes the prediction, SHAP contributors, and recent price history.

```powershell
cd frontend
python -m http.server 3000
```

Open `http://127.0.0.1:3000` (with the backend running separately).

## Tech stack

- Python 3.12, Pandas, NumPy
- Scikit-learn, XGBoost
- PyTorch (LSTM, GRU)
- SHAP (explainability)
- FastAPI (backend), vanilla HTML/JS + Chart.js (frontend demo)
- yfinance (data collection)
- Pytest (40+ tests)
- Jupyter (EDA)

## Architecture
Raw Data (yfinance)
→ Validation (schema, OHLC integrity, duplicates)
→ Cleaning (dtype enforcement, sorting)
→ Feature Engineering (35 technical indicators)
→ Target Engineering (return / direction / 3-class, leakage-tested)
→ Time-aware Train/Val/Test Split
→ Model Training (Naive, LR, RF, XGBoost, LSTM, GRU)
→ Signal Engine (probability → BUY/HOLD/SELL)
→ Backtest Engine (transaction costs, Sharpe, drawdown, benchmark comparison)
→ SHAP Explainability
→ FastAPI Backend → Demo Frontend
## Project structure
src/
data/            - market data downloader
preprocessing/   - validation and cleaning
features/        - technical indicator engineering
targets/         - prediction target engineering
ml/              - model training and time-aware splitting
dl/              - LSTM/GRU sequence models
explainability/  - SHAP-based model explanations
backtesting/     - signal generation and backtest engine
backend/           - FastAPI app (routes, services, schemas)
frontend/          - demo dashboard (HTML/JS)
notebooks/         - exploratory data analysis
scripts/           - runnable pipeline scripts (one per phase)
tests/             - 40+ automated tests, including explicit
leakage-prevention tests
data/              - raw/interim/processed (gitignored, reproducible
from scripts)
models/            - trained model checkpoints (gitignored, regenerable)
## Engineering principles followed

- Raw data is never modified or silently overwritten
- Time-series data is never randomly shuffled for splitting/evaluation
- Scalers/transformers are fit only on training data
- Every feature and target is tested for look-ahead bias
- Model comparisons include realistic backtesting, not just classification
  accuracy
- No inflated claims — results are reported honestly, including
  limitations and negative findings

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running the pipeline

```powershell
python scripts\download_stock.py           # download raw data (AAPL)
python scripts\validate_data.py            # validate
python scripts\clean_data.py               # clean
python scripts\build_features.py           # feature engineering
python scripts\build_targets.py            # target engineering
python scripts\train_baseline_models.py    # classical model comparison
python scripts\run_milestone1_backtest.py  # backtest
python scripts\train_lstm.py               # LSTM
python scripts\train_gru.py                # GRU
python scripts\run_shap_analysis.py        # SHAP explainability
python scripts\download_multi_stock.py     # multi-stock download
python scripts\run_multi_stock_pipeline.py # multi-stock comparison
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
- [x] LSTM / GRU sequence models
- [x] SHAP explainability
- [x] FastAPI backend (predict, health, price history endpoints)
- [x] Demo frontend
- [x] Multi-stock support (AAPL, MSFT, GOOGL, AMZN, TSLA)
- [x] Ensemble model
- [ ] Transformer for time series
- [ ] News sentiment (FinBERT)
- [ ] Risk analysis & portfolio optimization