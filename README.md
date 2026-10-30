# SPGT: Sectoral Patch Gated Transformer for daily sectoral CO2 emissions in China

Code, data and trained weights for the paper
"Holiday-aware forecasting of China's daily sectoral CO2 emissions around Chinese New Year".

SPGT forecasts the next 30 days of emissions for six sectors (Domestic Aviation, Ground Transport,
Industry, International Aviation, Power, Residential) from the previous 90 days. The model

1. splits each sector series into overlapping patches (length 16, stride 8) and applies temporal
   self-attention within each sector;
2. projects the calendar features of the forecast horizon into one embedding per patch and adds it
   to every patch token;
3. mixes information across sectors with a sectoral attention layer whose contribution is controlled
   by a learnable gate; the gate is set to zero whenever the forecast horizon contains a day of the
   Chinese New Year (CNY) window (day -7 to day +15).

## Repository layout

```
spgt/            model, data preparation, training loop, metrics
configs/         hyperparameters of the final model
scripts/         train.py, evaluate.py
data/            Carbon Monitor China daily sector data and official holiday table
weights/         trained weights for seeds 0-4
```

## Installation

Python 3.11.

```bash
pip install -r requirements.txt
```

A CUDA GPU is used if available; evaluation also runs on CPU.

## Data

`data/carbonmonitor_china_daily.csv` holds daily CO2 emissions (Mt CO2/day) for China from
2019-01-01 to 2026-03-31, taken from Carbon Monitor. `data/cn_official_holidays.csv` lists statutory
holidays and adjusted working days in China for 2019-2026. See `data/README.md` for sources.

Splits are chronological:

| split      | windows | target days               |
|------------|---------|---------------------------|
| train      | 1707    | 2019-04-01 to 2023-12-31  |
| validation | 338     | 2023-12-31 to 2024-12-31  |
| test       | 427     | 2024-12-31 to 2026-03-31  |

Each window has a 90-day input and a 30-day target. The input of the first validation (test)
window ends one day before the training (validation) cut-off, so the first target day of each later
split coincides with the last day of the preceding one.

Emissions are z-scored per sector with training-period statistics. Each input step carries 11
calendar features: sine/cosine of month, day of week and day of year; the CNY-window indicator;
days before and after the nearest CNY (divided by 180); official holiday and adjusted workday flags.

## Training

```bash
python scripts/train.py --seed 0
```

Weights are written to `runs/spgt_s<seed>.pt`. Hyperparameters are read from
`configs/spgt_final.json` (selected by random search on the 2024 validation year): d_model 64,
2 heads, 1 layer, dropout 0.1, patch 16 / stride 8, initial gate 0.2, AdamW with learning rate 3e-3
and weight decay 1e-3, cosine schedule, batch size 64, at most 65 epochs with early stopping
(patience 20) on validation MSE. Training takes a few seconds per seed on a single GPU.

## Evaluation

```bash
python scripts/evaluate.py --weights-dir weights
```

Metrics are computed on the raw scale and averaged over the six sectors (WAPE is pooled over
sectors). CNY-window MAE uses only target days inside the CNY window.

Expected output with the released weights:

| seed | MAE    | RMSE   | WAPE (%) | CNY-window MAE |
|------|--------|--------|----------|----------------|
| 0    | 0.3584 | 0.4810 | 6.51     | 0.6531         |
| 1    | 0.3644 | 0.5112 | 6.62     | 0.6537         |
| 2    | 0.3462 | 0.4752 | 6.29     | 0.5503         |
| 3    | 0.3534 | 0.5031 | 6.42     | 0.6275         |
| 4    | 0.3964 | 0.5457 | 7.20     | 0.6220         |
| mean | 0.3638 ± 0.0194 | 0.5032 ± 0.0281 | 6.61 ± 0.35 | 0.6213 ± 0.0422 |

Standard deviations are over the five seeds (ddof = 1). RMSE is the mean of the per-sector RMSE.
Retraining with `scripts/train.py` on the same software versions reproduced the seed-0 weights
exactly; other GPUs or library versions may give small differences.

## Citation

```bibtex
@article{chen_spgt,
  title   = {Holiday-aware forecasting of China's daily sectoral CO2 emissions around Chinese New Year},
  author  = {Chen, Shuhao and others},
  journal = {},
  year    = {},
  doi     = {}
}
```

## License

Code: MIT (see `LICENSE`). Carbon Monitor data: CC BY 4.0.
