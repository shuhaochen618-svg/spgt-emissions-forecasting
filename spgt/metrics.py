import numpy as np

from .data import SECTORS


def sector_metrics(y_true, y_pred, mask=None):
    """Per-sector and overall metrics on the raw scale (Mt CO2/day).

    Overall MAE / MSE / RMSE / MAPE are means of the per-sector values;
    overall WAPE is sum|e| / sum|y| * 100 pooled over all sectors.
    """
    rows = {}
    abs_e_tot, abs_y_tot = 0.0, 0.0
    for s, sec in enumerate(SECTORS):
        t, p = y_true[..., s], y_pred[..., s]
        if mask is not None:
            t, p = t[mask], p[mask]
        e = t - p
        safe = np.where(t == 0, 1e-8, t)
        mse = np.mean(e ** 2)
        rows[sec] = dict(MAE=np.mean(np.abs(e)), MSE=mse, RMSE=np.sqrt(mse),
                         MAPE=np.mean(np.abs(e / safe)) * 100,
                         WAPE=np.sum(np.abs(e)) / np.sum(np.abs(t)) * 100)
        abs_e_tot += np.sum(np.abs(e))
        abs_y_tot += np.sum(np.abs(t))
    overall = {k: np.mean([rows[sec][k] for sec in SECTORS]) for k in ('MAE', 'MSE', 'RMSE', 'MAPE')}
    overall['WAPE'] = abs_e_tot / abs_y_tot * 100
    rows['Overall'] = overall
    return rows


def evaluate(y_true, y_pred, cny_mask):
    """Overall metrics on all test targets plus MAE on target days inside the CNY window.

    y_true, y_pred: (N, H, S) raw scale; cny_mask: (N, H) boolean.
    """
    res = dict(sector_metrics(y_true, y_pred)['Overall'])
    res['CNY_MAE'] = sector_metrics(y_true, y_pred, cny_mask)['Overall']['MAE']
    res['nonCNY_MAE'] = sector_metrics(y_true, y_pred, ~cny_mask)['Overall']['MAE']
    return res
