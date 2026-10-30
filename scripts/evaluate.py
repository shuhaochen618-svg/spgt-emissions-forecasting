"""Evaluate saved SPGT weights on the test period and average over seeds.

python scripts/evaluate.py --weights-dir weights --seeds 0 1 2 3 4
"""
import argparse
import json
import os
import sys

import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from spgt.data import DATA_CSV, HOLIDAY_CSV, NUM_CALENDAR, make_splits, to_raw  # noqa: E402
from spgt.metrics import evaluate  # noqa: E402
from spgt.model import build_spgt  # noqa: E402
from spgt.training import device, make_loaders, predict  # noqa: E402

COLS = ['MAE', 'MSE', 'RMSE', 'WAPE', 'CNY_MAE']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights-dir', default=os.path.join(ROOT, 'weights'))
    ap.add_argument('--seeds', type=int, nargs='+')
    ap.add_argument('--config', default=os.path.join(ROOT, 'configs', 'spgt_final.json'))
    ap.add_argument('--data', default=DATA_CSV)
    ap.add_argument('--holidays', default=HOLIDAY_CSV)
    ap.add_argument('--out', help='optional CSV with per-seed metrics')
    a = ap.parse_args()

    cfg = json.load(open(a.config))
    dc, tc = cfg['data'], cfg['training']
    seeds = a.seeds if a.seeds is not None else cfg['seeds']
    splits = make_splits(a.data, a.holidays, dc['train_end'], dc['val_end'], dc['lookback'], dc['horizon'])
    _, _, te = make_loaders(splits, tc['batch_size'], tc['eval_batch_size'])
    _, Xf, y, dates = splits['test']
    sc = splits['scalers']
    y_raw, cny_mask = to_raw(y, sc), Xf[:, :, 4] > 0.5
    last = dates[-1] + pd.Timedelta(days=dc['horizon'] - 1)
    print(f'test windows: {len(dates)}, target days {dates[0]:%Y-%m-%d} to {last:%Y-%m-%d}')

    rows = []
    for s in seeds:
        model = build_spgt(cfg['model'], dc['lookback'], dc['horizon'], NUM_CALENDAR).to(device)
        state = torch.load(os.path.join(a.weights_dir, f'spgt_s{s}.pt'), map_location=device, weights_only=True)
        model.load_state_dict(state)
        m = evaluate(y_raw, to_raw(predict(model, te), sc), cny_mask)
        rows.append(dict(seed=s, **{k: m[k] for k in COLS}))

    df = pd.DataFrame(rows)
    with pd.option_context('display.float_format', '{:.4f}'.format):
        print(df.to_string(index=False))
    mean, std = df[COLS].mean(), df[COLS].std(ddof=1).fillna(0.0)
    print('mean +- std over seeds (ddof=1): ' + ', '.join(f'{k} {mean[k]:.4f} +- {std[k]:.4f}' for k in COLS))
    if a.out:
        df.to_csv(a.out, index=False)


if __name__ == '__main__':
    main()
