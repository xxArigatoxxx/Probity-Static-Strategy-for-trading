# Probity-Static-Strategy-for-trading

A quantitative, real-time automated trading framework for **XAUUSD (Gold)** on the 5-minute (M5) timeframe. This system combines **Digital Signal Processing (DSP)**, **Information Theory**, **GPU-accelerated Machine Learning (XGBoost + ONNX Runtime)**, and multi-layered **Risk Guardrails** directly integrated with **MetaTrader 5 (MT5)**.

---

## 🌟 Key Features

* **Advanced DSP & Chaos Feature Engineering**: Bypasses traditional lagging indicators. Uses 1D Kalman Filters (market noise extraction), Hilbert Transform (instantaneous wave phase angle), Order Flow Imbalance (OFI), Signal-to-Noise Ratio (SNR), and Permutation Entropy (Shannon entropy for market chaos measurement).
* **High-Performance In-Memory Data Pipeline**: Leverages **Polars** parallelized across multi-core CPUs (configured for 12 threads) and stores tick historical data in compressed `.parquet` format (Zstandard) on local SSD.
* **Leak-Free Model Validation**: Implements **Purged Walk-Forward Cross-Validation** (5 folds with a 12-bar purge window) to prevent look-ahead bias and correlation leakages in financial time series.
* **Probability Calibration (Temperature Scaling)**: Post-processes XGBoost raw logits using Temperature Scaling ($P_{\text{calibrated}} = \text{Sigmoid}(z / T)$ via L-BFGS-B optimization) to eliminate tree-model overconfidence.
* **GPU-Accelerated Real-Time Inference**: Converts trained XGBoost models to `.onnx` for microsecond-level runtime evaluation on NVIDIA GPUs (`CUDAExecutionProvider`).
* **Multi-Tiered Safety & Circuit Breakers**:
  * **Spread Guard**: Ignores trades if current spread expands $> 2\times$ above the 20-period moving average.
  * **Volatility Circuit Breaker**: Disables trading during severe market spikes ($> 95\text{th}$ percentile realized volatility).
  * **Entropy Circuit Breaker**: Blocks signals during high market chaos ($\text{Permutation Entropy} > 0.92$).
  * **Position Limits & Slippage Protection**: Enforces maximum 1 active position, strict Stop Loss / Take Profit limits, and Immediate-or-Cancel (`IOC`) execution filling with capped price deviation.

---

## 🏗 System Architecture

```text
                 [ MetaTrader 5 Terminal ]
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
   [ Historical Data ]               [ Live M5 Bar Feed ]
            │                                 │
            ▼                                 ▼
    [ DataEngine ]                     [ DataEngine ]
  (Polars + Scipy DSP)              (Real-time Window DSP)
            │                                 │
            ▼                                 ▼
   [ ModelEngine ]                   [ InferenceEngine ]
 ┌──────────────────────┐          ┌──────────────────────┐
 │ Purged Walk-Forward  │          │ Spread Guard         │
 │ XGBoost (CUDA)       │          │ Volatility Breaker   │
 │ Temp Scaling (T)     │          │ Entropy Breaker      │
 │ ONNX Export          │          │ ONNX CUDA Inference  │
 └──────────┬───────────┘          └──────────┬───────────┘
            │                                 │
            └─────────────► [ ONNX ] ─────────┘
                              Model
                                │
                                ▼
                       [ ExecutionEngine ]
                                │
                                ▼
                      [ MT5 Order Execution ]
```

---

## 🛠 Tech Stack

* **Language**: Python 3.10+
* **Trading Platform**: MetaTrader 5 (MT5 Python API)
* **Data Processing**: Polars, NumPy, SciPy (Signal processing)
* **Storage**: PyArrow / Parquet (`zstd` compression)
* **Machine Learning**: XGBoost, Scikit-learn, Scipy (`optimize.minimize`)
* **Inference**: ONNX, `onnxmltools`, ONNX Runtime (`CUDAExecutionProvider`)

---

## 📁 Repository Structure

```text
.
├── Main.py                 # Pipeline entry point & real-time M5 bar monitoring loop
├── data_engine.py          # MT5 connection, Parquet I/O, & DSP feature extraction
├── model_engine.py         # Purged Walk-Forward CV, XGBoost CUDA training & ONNX export
├── inference_engine.py     # ONNX CUDA runtime inference, probability calibration & guardrails
├── execution_engine.py    # Trade execution, position guard, SL/TP calculation & MT5 order handling
├── data/                   # Storage directory for .parquet market data
└── models/                 # Storage directory for exported .onnx models & JSON metadata
```

---

## 📊 Feature Set Overview

The model relies on **11 engineered features** derived from price action, signal processing, and order flow:

| Feature | Category | Description |
| :--- | :--- | :--- |
| `log_return` | Statistical | Stationary price change ($\ln(P_t / P_{t-1})$) |
| `realized_vol` | Volatility | 20-period rolling standard deviation of log returns |
| `ofi_1h` | Order Flow | 1-hour accumulated Order Flow Imbalance (Tick Rule / BVC) |
| `ofi_zscore` | Order Flow | Normalized z-score of raw Order Flow Imbalance |
| `kalman_residual` | DSP | Difference between actual market price and Kalman equilibrium state |
| `kalman_zscore` | DSP | Normalized z-score of Kalman filter residuals |
| `hilbert_phase_deg`| DSP | Instantaneous phase angle ($0^\circ$ – $360^\circ$) via Hilbert Transform |
| `hilbert_sin` | DSP | Sine component of phase angle for continuous cyclic representation |
| `hilbert_cos` | DSP | Cosine component of phase angle for continuous cyclic representation |
| `perm_entropy` | Information | Normalized Shannon Entropy measuring market disorder/complexity |
| `snr_db` | Signal | Signal-to-Noise Ratio in decibels (dB) |

---

## ⚙️ Execution & Risk Guardrails

```text
                          [ Incoming Closed Candle ]
                                     │
                                     ▼
                      [ Spread Guard Check ] ──(Spread > 2x MA20)──► [ SKIP ]
                                     │
                                     ▼
                  [ Volatility Circuit Breaker ] ──(Vol > 95th Pct)──► [ SKIP ]
                                     │
                                     ▼
                   [ Entropy Circuit Breaker ] ──(Entropy > 0.92)──► [ SKIP ]
                                     │
                                     ▼
                     [ ONNX Model GPU Prediction ]
                                     │
                                     ▼
                    [ Temperature Scaling Calibration ]
                                     │
                                     ▼
                 [ Min Win Probability Threshold ] ──(P(Win) < 0.62)──► [ SKIP ]
                                     │
                                     ▼
                    [ DSP Phase Angle Confirmation ]
                       ├─ SELL: Z > 1.2 & Phase in [45°, 165°]  ──► [ EXECUTE SELL ]
                       ├─ BUY:  Z < -1.2 & Phase in [225°, 345°] ──► [ EXECUTE BUY ]
                       └─ Neutral / Mid-range                   ──► [ SKIP ]
```

---

## ⚠️ Disclaimer

> **Note:** This system is built for quantitative research and automated trading demonstration purposes. Financial trading involving gold and levered instruments carries high risk. Test thoroughly on demo accounts before considering live market deployment.
