# SPGT: Spatiotemporal Patch Graph Transformer

Official implementation for the manuscript:

**Forecasting Lunar New Year disruptions in China's daily sectoral CO2 emissions for near-real-time monitoring**

This repository provides the implementation of the holiday-aware Spatiotemporal Patch Graph Transformer (SPGT) for 30-day forecasting of China's sectoral daily CO2 emissions using the Carbon Monitor dataset.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

<p align="center">
  <img src="Figure1.png" alt="Graphical abstract" width="100%">
</p>

---

## Overview

Near-real-time CO2 emission estimates are useful for environmental monitoring and short-term assessment, but daily emissions contain strong calendar effects. In China, the Lunar New Year holiday shifts on the Gregorian calendar and produces recurring but date-shifting disruptions in industrial and transport activity.

SPGT addresses this short-term forecasting problem with three components:

- patch-based temporal representation for daily multivariate emission sequences;
- future calendar covariates for holiday-aware forecasting;
- sector-attention layers for modelling predictive associations among emission sectors.

The sector-attention outputs should be interpreted as exploratory predictive associations, not as causal evidence of physical sectoral coupling.

---

## Repository structure

```text
.
├── LICENSE
├── README.md
├── requirements.txt
├── setup.py
├── data/
│   └── carbonmonitor-global_datas_2026-05-22.csv
├── spgt/
│   ├── __init__.py
│   ├── models.py
│   └── preprocess.py
└── examples/
    ├── inference_demo.py
    └── train_eval.py
```

---

## Installation

```bash
git clone https://github.com/shuhaochen618-svg/spgt-emissions-forecasting.git
cd spgt-emissions-forecasting
pip install -r requirements.txt
pip install -e .
```

---

## Quickstart

### Minimal inference demo

```bash
python examples/inference_demo.py
```

### Training and evaluation example

```bash
python examples/train_eval.py --data_path "data/carbonmonitor-global_datas_2026-05-22.csv" --epochs 15 --batch_size 32 --seed 42
```

The example uses a 90-day lookback window and a 30-day forecasting horizon. By default, it uses a chronological split with training targets ending on 31 December 2023, validation targets ending on 31 December 2024, and test targets thereafter.

---

## Manuscript benchmark results

The following raw-scale results correspond to the benchmark table reported in the current manuscript for China's 2025-2026 test period and a 30-day forecasting horizon.

| Model | MAE (Mt CO2/day) | MSE | MAPE (%) |
| :--- | ---: | ---: | ---: |
| SPGT (Ours) | **0.4074** | **0.6283** | **47.57** |
| SPGT (w/o CNY) | 0.4691 | 0.8445 | 51.30 |
| DLinear | 0.4656 | 0.8458 | 64.70 |
| Linear | 0.5426 | 1.0865 | 79.71 |
| GRU | 0.5775 | 1.0820 | 144.35 |
| RNN | 0.5816 | 1.0876 | 232.34 |
| LSTM | 0.6002 | 1.1488 | 187.82 |
| MLP | 0.6008 | 1.5362 | 52.03 |
| Transformer | 0.6022 | 1.3902 | 53.81 |
| Random Forest | 0.6215 | 1.5260 | 60.37 |

The MAPE values should be interpreted carefully because sector-level daily emissions can contain small denominators, especially for low-emission sectors.

---

## Data note

The daily CO2 emissions data analysed in the manuscript were obtained from the publicly available Carbon Monitor dataset. Carbon Monitor is a living dataset and may be updated or revised over time; therefore, reported results correspond to the data release used for the present analysis.

---

## Scope of the repository

This repository supports short-term emission forecasting and reproducibility of the SPGT modelling workflow. It does not provide a policy-grade national emissions-peak estimate, and recursive long-horizon projections should be treated only as exploratory model outputs.

---

## License

This project is released under the MIT License.
