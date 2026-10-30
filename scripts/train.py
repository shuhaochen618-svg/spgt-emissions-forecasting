"""Train SPGT with one seed and save the best-validation weights.

python scripts/train.py --seed 0
"""
import argparse
import json
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from spgt.data import DATA_CSV, HOLIDAY_CSV, NUM_CALENDAR, make_splits, to_raw  # noqa: E402
from spgt.metrics import evaluate  # noqa: E402
from spgt.model import build_spgt  # noqa: E402
from spgt.training import device, make_loaders, predict, set_seed, train_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--config', default=os.path.join(ROOT, 'configs', 'spgt_final.json'))
    ap.add_argument('--data', default=DATA_CSV)
    ap.add_argument('--holidays', default=HOLIDAY_CSV)
    ap.add_argument('--out-dir', default=os.path.join(ROOT, 'runs'))
    ap.add_argument('--verbose', action='store_true')
    a = ap.parse_args()

    cfg = json.load(open(a.config))
    dc, tc = cfg['data'], cfg['training']
    splits = make_splits(a.data, a.holidays, dc['train_end'], dc['val_end'], dc['lookback'], dc['horizon'])

    t0 = time.time()
    set_seed(a.seed)
    tr, va, te = make_loaders(splits, tc['batch_size'], tc['eval_batch_size'])
    model = build_spgt(cfg['model'], dc['lookback'], dc['horizon'], NUM_CALENDAR).to(device)
    model, val_mse, n_ep = train_model(model, tr, va, tc['max_epochs'], tc['lr'], tc['weight_decay'],
                                       tc['patience'], a.verbose)

    os.makedirs(a.out_dir, exist_ok=True)
    path = os.path.join(a.out_dir, f'spgt_s{a.seed}.pt')
    torch.save(model.state_dict(), path)

    _, Xf, y, _ = splits['test']
    sc = splits['scalers']
    m = evaluate(to_raw(y, sc), to_raw(predict(model, te), sc), Xf[:, :, 4] > 0.5)
    print(f'seed {a.seed}: epochs {n_ep}, val MSE {val_mse:.5f}, time {time.time() - t0:.1f}s')
    print(f'test MAE {m["MAE"]:.4f}  RMSE {m["RMSE"]:.4f}  WAPE {m["WAPE"]:.2f}%  CNY-window MAE {m["CNY_MAE"]:.4f}')
    print(f'saved {path}')


if __name__ == '__main__':
    main()
