# Sales & Demand Forecasting with Time Series Analysis
### Nielsen Book Sales Dataset | Cambridge Data Science Programme

A multi-model time series forecasting framework built for Nielsen, 
comparing classical statistical, machine learning, deep learning and 
hybrid approaches to predict book sales and assess publication 
longevity, applied to two contrasting titles: The Very Hungry 
Caterpillar and The Alchemist.

---

## Problem

Nielsen is developing a service for independent publishers to make 
data-driven decisions about new publications. 
The goal: accurately forecast post-publication sales profiles for better stock control, 
initial investment decisions and understanding of a title's economic 
lifespan.

---

## Dataset

- **UK Weekly Trended Timeline:** 227,224 rows × 14 columns of weekly 
  sales data
- **ISBN List:** 500 rows × 11 columns of book metadata
- **Focus titles:** The Very Hungry Caterpillar (stable, seasonal) 
  and The Alchemist (volatile, regime-shifting)

---

## Approach

Four model families compared across both titles:

| Model | Type | Key Parameters |
|-------|------|----------------|
| SARIMA | Classical Statistical | Auto ARIMA via AIC |
| XGBoost | Ensemble ML | Recursive multi-step, 32-week horizon |
| LSTM | Deep Learning | KerasTuner + early stopping |
| SARIMA-LSTM Sequential | Hybrid | SARIMA residuals fed into LSTM |
| SARIMA-LSTM Parallel | Hybrid | Weighted average, grid search |

---

## Key Results

### The Very Hungry Caterpillar

| Model | MAE | MAPE |
|-------|-----|------|
| SARIMA | 353.14 | 18.63% |
| XGBoost | 539.06 | 23.55% |
| LSTM | 614.47 | 34.12% |
| SARIMA-LSTM Sequential | 352.15 | 18.59% |
| **SARIMA-LSTM Parallel** | **339.93** | **17.86%** [Recommended] |

Optimal weights: SARIMA=0.7, LSTM=0.3: SARIMA captures structural 
trend and seasonality while LSTM corrects residual nonlinearities.

### The Alchemist

| Model | MAE | MAPE |
|-------|------|------|
| SARIMA | 155.07 | 29.75% |
| XGBoost | 147.29 | 28.77% |
| LSTM | 258.95 | 44.98% |
| SARIMA-LSTM Sequential | 164.43 | 32.13% |
| **SARIMA-LSTM Parallel** | **133.26** | **26.68%** [Recommended] |

Equal weights (SARIMA=0.5, LSTM=0.5): LSTM's contribution increases 
for volatile, regime-shifting series where SARIMA alone misses 
nonlinear patterns.

---

## Key Findings

- The **parallel hybrid SARIMA-LSTM** is the best performing model 
  across both titles, combining SARIMA's structural baseline with 
  LSTM's local responsiveness
- For **stable seasonal series** (The Very Hungry Caterpillar), 
  SARIMA dominates (0.7 weight): LSTM provides dampening correction
- For **volatile regime-shifting series** (The Alchemist), equal 
  weighting is optimal: LSTM captures nonlinear patterns SARIMA misses
- **Weekly data outperforms monthly** for volatile series: monthly 
  aggregation discards the high-frequency information the model needs 
  to detect regime shifts
- For stable series, weekly and monthly MAPE are nearly identical 
  (24%): the noise-signal trade-off is fair

---

## Methods

```python
# Core libraries
statsmodels          # SARIMA, SARIMAX, ACF/PACF, ADF test
pmdarima             # Auto ARIMA
xgboost              # XGBoost recursive forecasting
tensorflow/keras     # LSTM architecture
keras_tuner          # Hyperparameter optimisation
sklearn              # Preprocessing, metrics
```

---

## Limitations

- Recursive multi-step forecasting causes amplitude damping in tree 
  and neural models over long horizons
- LSTM validation reused training data, risk of overfitting
- Optimal hybrid weights are dataset specific and require 
  re-optimisation for new titles

---

## Recommendation

Deploy the weekly parallel hybrid SARIMA-LSTM model for both stable 
and volatile titles. For structurally disrupted series (zero-sales 
breaks, unprecedented demand spikes), maintain conservative safety 
stock buffers regardless of model choice.

---

## Author

**Israel Olujobi**  
Cambridge Data Science & AI Programme, June 2026
