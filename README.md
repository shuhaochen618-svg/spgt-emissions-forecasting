# SPGT: Spatiotemporal Patch Graph Transformer

Official implementation of the **Spatiotemporal Patch Graph Transformer (SPGT)** (also referred to as **TSGT**) for multi-sectoral daily carbon emissions forecasting.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

---

## 📖 Introduction & Background

Accurate daily carbon emissions forecasting is essential for dynamic power grid dispatching and dynamic carbon trading market operations. However, high-frequency emission tracking poses two primary challenges:
1. **Shifting Cultural Holidays**: The Chinese New Year (CNY) holiday shifts annually on the Gregorian calendar by up to 20 days. This holiday causes massive shutdowns across manufacturing and industrial sectors, producing a severe 15-day emission "dip." Traditional models experience phase lags and spikes because they are holiday-blind.
2. **Cross-Sectoral Coupling**: Emissions across sectors (e.g., Power and Industry) are strongly coupled. Vanilla state-of-the-art time-series models like **PatchTST** treat variables as completely independent channels, failing to capture cross-channel dependency.

**SPGT** directly addresses these limits by extending PatchTST's channel-independent temporal patching backbone with:
* **CNY Patch-Embeddings**: Warping temporal representations by projecting calendar coordinates into the patch representation.
* **Decoupled Cross-Sector Graph Attention**: Utilizing multi-head self-attention across sectors to model carbon-flow couplings, gated dynamically by holiday-intensity vectors.

---

## 🏗️ Architecture Overview

```
                       [Input Window: L=90 days]
                                  │
       ┌───────────────────────────┴───────────────────────────┐
       ▼                                                       ▼
[Emissions: (B, 6, L)]                                [Calendar: (B, L, 5)]
       │                                                       │
   [Padding]                                               [Padding]
       │                                                       │
[Padded Emissions: (B, 6, 96)]                        [Padded Calendar: (B, 96, 5)]
       │                                                       │
   [Patching: unfold stride=8, len=16]                      [Patching & Flattening]
       │                                                       │
[Emissions Patches: (B, 6, N=11, P=16)]               [Calendar Patches: (B, N=11, 5*16)]
       │                                                       │
[Shared Patch Projection: Linear(P -> d)]             [Calendar Projection: Linear(5*P -> d)]
       └───────────────────────────┬───────────────────────────┘
                                   ▼
                     [Combined Patches: (B, 6, N, d)]
                                   │
                     [+ Positional & Sector Embeddings]
                                   │
                [PatchTST Backbone: Temporal Attention Layer]
                                   │
                      [Cross-Sector Graph Attention]
                                   │
                     [Temporal Decoder: Linear(N -> H)]
                                   │
                     [Output Projection: Linear(d -> 1)]
                                   ▼
                    [Output Predictions: (B, H=30, 6)]
```

---

## 📂 Repository Structure

```
.
├── LICENSE                  # MIT License
├── README.md                # Documentation and usage guide
├── requirements.txt         # Package dependencies
├── setup.py                 # Package installer script
├── spgt/                    # Core source library
│   ├── __init__.py          # Exposed classes and helpers
│   ├── models.py            # PyTorch model definitions
│   └── preprocess.py        # Holiday projection and dataset utilities
└── examples/                # Running examples
    ├── inference_demo.py    # Basic test pass with dummy tensors
    └── train_eval.py        # Complete training pipeline (includes synthetic data generator)
```

---

## ⚡ Installation

Clone the repository and install dependencies in editable mode:

```bash
git clone https://github.com/your-username/spgt.git
cd spgt
pip install -r requirements.txt
pip install -e .
```

---

## 🚀 Quickstart

### 1. Minimal Inference Demo
To check if the package works, run the inference demo script which computes a forward pass on dummy tensors:

```bash
python examples/inference_demo.py
```

Or write your own script:

```python
import torch
from spgt import TemporalSectoralTransformer

# 6 sectors, 5 calendar dimensions, 90 lookback days, 30 forecast days
model = TemporalSectoralTransformer(lookback=90, horizon=30, num_sectors=6, num_calendar=5)

X_hist = torch.randn(4, 90, 11)      # (batch, lookback, num_sectors + calendar_dims)
X_fut_cal = torch.randn(4, 30, 5)    # (batch, horizon, calendar_dims)

model.eval()
with torch.no_grad():
    y_pred, t_attn, s_attn = model(X_hist, X_fut_cal)

print("Predictions shape:", y_pred.shape)  # Expected: torch.Size([4, 30, 6])
```

### 2. Complete Training and Evaluation
You can train SPGT on synthetic daily data immediately by running:

```bash
python examples/train_eval.py --epochs 15 --batch_size 32
```

If you have downloaded the Carbon Monitor daily CSV dataset (e.g. `carbonmonitor-global_datas_2026-05-22.csv`), specify its location:

```bash
python examples/train_eval.py --data_path "/path/to/carbonmonitor-global.csv" --epochs 50
```

---

## 📊 Benchmarks (China Test Set)

Raw, unscaled benchmark results comparing SPGT with traditional recurrent networks, point-wise Transformers, and channel-independent baselines on the daily Carbon Monitor dataset for a 30-day horizon:

| Model | Overall MAE (Mt CO₂/day) | Overall MSE | Overall MAPE (%) | Status |
| :--- | :---: | :---: | :---: | :---: |
| **SPGT (Ours)** | **0.4321** | **0.6864** | **48.40%** | **Proposed** |
| **SPGT (w/o CNY)** | 0.4632 | 0.8174 | 57.42% | Ablation |
| **DLinear** | 0.4656 | 0.8458 | 64.70% | Baseline |
| **LSTM** | 0.6002 | 1.1488 | 187.82% | Recurrent |
| **Transformer** | 0.6022 | 1.3902 | 53.81% | Point-wise |

---

## ✍️ Citation

If you use this model or code in your academic work, please cite:

```bibtex
@article{spgt2026carbon,
  title={Capturing Lunar New Year Dynamics and Sectoral Coupling in China's Daily Carbon Emissions: A Patch-Based Spatiotemporal Graph Transformer Approach},
  author={Carbon Dynamics Research Team},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```
