# Probabilistic Demand Forecasting — Weekend Project

## Why this project
Prep for the Honda parts-forecasting project. The goal isn't point accuracy (MAE/MASE/WAPE) —
it's building real intuition for **evaluating the quality of a full predictive distribution**:
CRPS, calibration, and pinball loss. This maps directly to inventory decisions: knowing your
P10/P50/P90 demand, not just your expected demand, is what tells you how much safety stock to hold.

## Dataset
**M5 Forecasting (Uncertainty track framing)** — Walmart daily sales, ~5.4 years, CA/TX/WI stores,
3 categories (Hobbies, Household, Foods). Using the *task* from the Uncertainty track (predict
quantiles, evaluate with pinball loss) but scoped way down from competition scale.

- Source: Kaggle "M5 Forecasting - Uncertainty" (or Accuracy — same base data, just use `sales_train_validation.csv`, `calendar.csv`, `sell_prices.csv`)
- Scope: **20–30 SKUs only**, hand-picked to include:
  - A few high-volume, steady-demand items
  - A few intermittent/spiky items (lots of zero days) — closer to real parts demand
- Level: item-level only. Skip store/category/total aggregation (that's a Uncertainty-track-scale concern, not needed here).
- Horizon: 28-day forecast (matches M5 convention, arbitrary but standard)
- Quantiles: use 5, not all 9 — **{0.1, 0.25, 0.5, 0.75, 0.9}**. Enough to see tail behavior without 2x the training runs.

## Model zoo (scoped for the weekend)
| Tier | Model | Approach |
|---|---|---|
| Tier 1 | Exponential smoothing / naive | Point forecast → quantiles from empirical residual distribution |
| Tier 2 | LightGBM | Point forecast (lag + rolling + calendar features) → quantiles from residuals, **or** native `objective='quantile'` for 2–3 key quantiles if time allows |

(ARIMA/SARIMA, XGBoost, CatBoost, Prophet, TFT, N-BEATS/N-HiTS are the fuller model zoo for
the actual Honda project — out of scope this weekend. This project is about the *evaluation
methodology*, which transfers to any of them.)

## Evaluation metrics (the actual point of this project)
1. **Pinball loss** — per-quantile, per-model. Are certain quantiles (e.g. the tails) worse than others?
2. **CRPS** — one overall distributional accuracy number per model. Use `properscoring` or `scoringrules` package.
3. **Calibration / PIT histogram** — for each observation, compute where the true value falls in the predicted CDF. A well-calibrated model gives a ~uniform PIT histogram. Systematic skew = over/under-confidence.

Explicitly **not** the focus: MAE/MASE/WAPE. Compute them only as a side sanity check if there's time, not as headline results.

## Plan

### Saturday evening
- [ ] Download M5 data, filter to 20–30 chosen SKUs
- [ ] EDA notebook: plot series, check intermittency %, seasonality, calendar events
- [ ] Tier 1 baseline: exponential smoothing point forecast + residual-based quantiles
- [ ] Build eval pipeline: pinball loss fn, CRPS fn, PIT histogram fn — run end-to-end on baseline
- **Stop-point goal:** working baseline + working eval pipeline (all 3 metrics computing correctly)

### Sunday
- [ ] LightGBM point forecast: lag features, rolling mean/std, day-of-week, calendar/event flags
- [ ] Convert to quantiles (residual method, or native quantile objective for 3 quantiles if time allows)
- [ ] Re-run eval pipeline on LightGBM predictions
- [ ] Comparison: side-by-side PIT histograms (baseline vs. LightGBM), pinball loss table across quantiles
- [ ] Write up 3–5 sentences of findings (e.g., "LightGBM tightened median error but was overconfident at tails")

## Repo structure
```
parts-forecasting-practice/
├── README.md                  # short project summary, mirrors this plan
├── data/                      # raw + filtered M5 subset (gitignored if large)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_expsmoothing.ipynb
│   ├── 03_lightgbm_quantile.ipynb
│   └── 04_comparison.ipynb
├── src/
│   ├── data_loader.py         # load + filter M5 subset
│   ├── features.py            # lag/rolling/calendar feature engineering
│   ├── models.py               # baseline + LightGBM training functions
│   └── evaluation.py          # pinball_loss(), crps(), pit_histogram()
├── requirements.txt
└── PLAN.md                    # this file
```

## Outcomes (fill in after)
- Baseline pinball loss (per quantile):
- LightGBM pinball loss (per quantile):
- Baseline CRPS:
- LightGBM CRPS:
- Calibration observations:
- What would change for the real Honda project (multi-SKU scale, cost-asymmetric loss for under- vs over-ordering, etc.):