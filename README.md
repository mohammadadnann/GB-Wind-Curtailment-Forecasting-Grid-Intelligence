# GB Wind Curtailment Forecasting and Grid Intelligence

A day ahead forecasting platform for Scottish wind curtailment at the B6 transmission boundary. I built this end to end from raw Elexon API data through to a deployed FastAPI service and Streamlit dashboard to demonstrate what a production grade energy forecasting system looks like in practice.

---

## What problem does this solve?

Scotland generates more wind power than the B6 transmission boundary can export south. When this happens, the National Energy System Operator instructs wind farms to reduce their output. This is called curtailment. In 2025 it cost the GB system an estimated £500m to £1bn in constraint payments, and the problem is growing as offshore wind capacity outpaces grid upgrades.

A day ahead forecast of when curtailment will happen and how much helps:
- Battery operators decide when to charge from excess wind energy
- Grid operators pre position flexible assets before constraints bind
- Traders price constraint risk into day ahead positions

---

## Dashboard

<img width="1440" height="789" alt="Screenshot " src="https://github.com/user-attachments/assets/e63542ee-7b3b-4e61-9135-1accf124ed4c" />


---

## Data sources

| Source | What I used it for | Coverage |
|--------|--------------------|----------|
| Elexon Insights API | Bid offer acceptances, physical notifications, bid offer data | April 2023 to March 2026 |
| Open Meteo historical forecast API | Wind speed and direction at 100m hub height | April 2023 to March 2026 |
| NESO day ahead constraint flows and limits | B6, SCOTEX and NKILGRMO boundary limits | April 2023 to March 2026 |
| REPD (DESNZ) | Wind farm coordinates and installed capacity | Q2 2026 extract |

---

## How I derived curtailment

Curtailment is not published anywhere so I had to calculate it from scratch.

For each Scottish wind unit I:
1. Reconstructed what the unit planned to generate from its Physical Notifications
2. Reconstructed what NESO actually instructed it to generate from Bid Offer Acceptances
3. Calculated the gap minute by minute and summed it to settlement periods

The tricky part was that NESO reissues instructions roughly every 20 minutes, each one superseding the last. Reading the rows naively would show ramp ups that never happened and badly understate curtailment. I fixed this by processing acceptances in issuance time order and letting later ones overwrite earlier ones at each minute.

I also fixed a BST timezone bug where the minute grid was built on UTC calendar days instead of local settlement days. This silently dropped the last hour of every summer day. After the fix I added tests covering the 46 period spring clock change day, the normal 48 period day, and the 50 period autumn clock change day.

---

## Features

All 24 features are verified against the prediction contract: the forecast for day D is issued at 11:00 UTC on D-1.

| Feature | Source | Available at cutoff |
|---------|--------|-------------------|
| curtailment_lag_2d | Derived target, shift(96) on half hourly series | Yes |
| curtailment_lag_7d | Derived target, shift(336) | Yes |
| curtailment_roll_mean_7d | Rolling mean ending at D-2 | Yes |
| constrained_roll_mean_7d | Rolling constraint frequency ending at D-2 | Yes |
| Calendar features (month, hour, day of week, season) | Forecast timestamp | Yes |
| b6_limit_mw, b6_limit_vs_30d_median, b6_outage_flag | NESO constraint limits | Yes |
| scotex_limit_mw, scotex_limit_vs_30d_median, scotex_outage_flag | NESO constraint limits | Yes |
| nkilgrmo_limit_mw, nkilgrmo_limit_vs_30d_median, nkilgrmo_outage_flag | NESO constraint limits | Yes |
| wind_speed_100m, wind_direction_sin, wind_direction_cos, wind_gust_10m | Open Meteo forecast archive | Approximate |
| avg_bid_price_gbp_mwh | BOD bid prices joined to curtailment target | Yes (cost model only) |

I used shift(96) for D-2, not shift(48). On a half hourly series, shift(48) gives D-1 which leaks same day information into the features. I added timestamp based leakage tests to catch this.

---

## Model

I used a two stage hurdle model:

**Stage 1:** A LightGBM classifier that predicts whether a period will have any curtailment at all.

**Stage 2:** A LightGBM regressor that predicts how much curtailment in MWh, trained only on periods that were actually curtailed.

The final prediction combines both: predicted MWh = P(constrained) x predicted volume if constrained.

I log transform the volume target before training to handle the long tail of very large curtailment events.

For uncertainty I trained quantile regressors at P10, P50 and P90 using a zero inflated mixture approach. I then applied conformal calibration on a held out calibration slice to widen the bands until they achieved the target coverage.

I also built a cost model that predicts constraint payment cost in pounds per period, using BOD bid prices joined to the curtailment target as the training label.

---

## Results

| Model | MAE (MWh) | RMSE (MWh) | Skill vs persistence |
|-------|-----------|------------|----------------------|
| Hurdle LightGBM | **347.3** | **525.2** | **+44.6%** |
| Zero baseline | 572.5 | 881.2 | +8.6% |
| Persistence D-2 | 626.7 | 866.8 | 0.0% |
| Weekly D-7 | 704.1 | 949.6 | -12.4% |
| Physical heuristic | 1381.5 | 2473.2 | -120.4% |

All metrics from rolling origin expanding window backtesting across 6 folds, one calendar quarter per test set.

**Quantile interval coverage** (mean across 6 folds):
- P10 to P90 coverage: 78.1% against an 80% target
- Mean conformal widening factor: 1.825

**Cost model** (mean across 6 folds):
- Mean MAE: £11,697 per period
- Total actual across test folds: £497m
- Total predicted: £335m
- Consistent mild under prediction rather than erratic error
- 3 year total curtailment cost estimated at £898m from BOD bid prices (~£299m/year)

**Battery dispatch** (50 MW battery, mean across 6 folds):
- Model: £43,329/MW/year
- Persistence: £38,666/MW/year
- Perfect foresight ceiling: £39,979/MW/year
- Note: individual fold inversions are a known battery capacity mechanic where clustered optimal picks hit the energy ceiling earlier than spread out picks. Disclosed rather than hidden.

**SHAP findings:**
- wind_speed_100m dominates both the classifier and volume regressor by a wide margin, consistent with the physics of curtailment
- scotex_limit_vs_30d_median ranks 3rd in the classifier with SHAP importance 0.30, above B6 at 0.20, showing the SCOTEX boundary carries more independent signal for constraint prediction than the widely cited B6 boundary
- nkilgrmo_limit_vs_30d_median ranks 8th in the classifier at 0.14, confirming northern boundaries add signal beyond B6 alone
- Local SHAP on the worst single error (2,976 MWh under prediction on 2026-01-12) showed nkilgrmo_limit_mw pushing the prediction down despite high wind speed at 15.6 m/s, tracing the miss to the model under-weighting the northern boundary during an extreme storm

**Fleet and data verified:**
- 97 Scottish transmission wind units, 11,608 MW total capacity
- Approximately 12 million raw rows across BOALF, PN and BOD datasets
- 525,994 curtailed unit periods, 21.96 TWh total across April 2023 to March 2026
- 22 unit tests passing including spring and autumn clock change cases
- 5 timestamp based leakage tests passing

---


## Limitations

The API and dashboard serve historical backtest dates only from April 2023 to March 2026. They do not construct live day ahead features from current data sources. A live system would need a nightly job that fetches tomorrow's weather, constraint limits and recent settlement data at 11:00 UTC on D-1.

Weather forecast vintage is approximated. The Open Meteo archive does not expose which NWP model run produced each value. We assume the D-1 06:00 UTC model run as the source.

The cost model is an estimate based on the bid payment leg only, not the full replacement cost figure, and uses standing BOD prices rather than the actual accepted prices from settlement.

The battery simulation has a known inversion mechanic where individual folds may show model exceeding perfect foresight due to battery capacity dynamics. The aggregate across all folds is the meaningful metric.
