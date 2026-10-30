import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CSV = os.path.join(ROOT, 'data', 'carbonmonitor_china_daily.csv')
HOLIDAY_CSV = os.path.join(ROOT, 'data', 'cn_official_holidays.csv')

LOOKBACK, HORIZON = 90, 30
TRAIN_END, VAL_END = '2023-12-31', '2024-12-31'

SECTORS = ['Domestic Aviation', 'Ground Transport', 'Industry', 'International Aviation', 'Power', 'Residential']

CNY_DATES = {
    2018: '2018-02-16', 2019: '2019-02-05', 2020: '2020-01-25', 2021: '2021-02-12',
    2022: '2022-02-01', 2023: '2023-01-22', 2024: '2024-02-10', 2025: '2025-01-29',
    2026: '2026-02-17', 2027: '2027-02-06', 2028: '2028-01-26', 2029: '2029-02-13',
    2030: '2030-02-03', 2031: '2031-01-23',
}

# CNY window: day -7 to day +15 relative to Lunar New Year's Day
CNY_WINDOW = (-7, 15)

CALENDAR_COLS = ['month_sin', 'month_cos', 'dow_sin', 'dow_cos', 'holiday_active', 'doy_sin', 'doy_cos',
                 'pre_cny', 'post_cny', 'official_holiday', 'adjusted_workday']
NUM_CALENDAR = len(CALENDAR_COLS)


def cny_distance(date):
    """Signed number of days from date to the nearest Lunar New Year's Day."""
    dt = pd.to_datetime(date)
    options = [pd.to_datetime(CNY_DATES[y]) for y in (dt.year - 1, dt.year, dt.year + 1) if y in CNY_DATES]
    nearest = min(options, key=lambda d: abs((dt - d).days))
    return (dt - nearest).days


def load_emissions(csv_path=DATA_CSV, country='China'):
    """Daily sector emissions (Mt CO2/day) as a date-indexed frame with CNY columns."""
    df = pd.read_csv(csv_path)
    df = df[df['country'] == country].copy()
    df['date_parsed'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
    df = df.sort_values('date_parsed')
    df = df.pivot(index='date_parsed', columns='sector', values='value').ffill().bfill()
    df['month'] = df.index.month
    df['day_of_week'] = df.index.dayofweek
    df['day_of_year'] = df.index.dayofyear
    df['holiday_dist'] = [cny_distance(d) for d in df.index]
    lo, hi = CNY_WINDOW
    df['holiday_active'] = ((df['holiday_dist'] >= lo) & (df['holiday_dist'] <= hi)).astype(float)
    return df


def official_flags(index, holiday_csv=HOLIDAY_CSV):
    h = pd.read_csv(holiday_csv, parse_dates=['date'])
    off = set(h.loc[h.type == 'holiday', 'date'])
    work = set(h.loc[h.type == 'workday', 'date'])
    return (np.array([d in off for d in index], dtype=float),
            np.array([d in work for d in index], dtype=float))


def calendar_features(df, holiday_csv=HOLIDAY_CSV):
    """The 11 calendar features in CALENDAR_COLS order."""
    m, w, y = df['month'].values, df['day_of_week'].values, df['day_of_year'].values
    dist = df['holiday_dist'].values.astype(float)
    oh, aw = official_flags(df.index, holiday_csv)
    cols = {
        'month_sin': np.sin(2 * np.pi * (m - 1) / 12), 'month_cos': np.cos(2 * np.pi * (m - 1) / 12),
        'dow_sin': np.sin(2 * np.pi * w / 7), 'dow_cos': np.cos(2 * np.pi * w / 7),
        'holiday_active': df['holiday_active'].values,
        'doy_sin': np.sin(2 * np.pi * (y - 1) / 365.25), 'doy_cos': np.cos(2 * np.pi * (y - 1) / 365.25),
        'pre_cny': np.clip(-dist, 0, None) / 180.0, 'post_cny': np.clip(dist, 0, None) / 180.0,
        'official_holiday': oh, 'adjusted_workday': aw,
    }
    return pd.DataFrame(cols, index=df.index)[CALENDAR_COLS]


def windows(em, cal, dates, lookback, horizon):
    """Sliding windows: X_hist (N, L, S + C), X_fut_cal (N, H, C), y (N, H, S), first target dates."""
    Xh, Xf, y, d = [], [], [], []
    for i in range(len(em) - lookback - horizon + 1):
        Xh.append(np.hstack([em[i:i + lookback], cal[i:i + lookback]]))
        Xf.append(cal[i + lookback:i + lookback + horizon])
        y.append(em[i + lookback:i + lookback + horizon])
        d.append(dates[i + lookback])
    return np.array(Xh), np.array(Xf), np.array(y), d


def make_splits(csv_path=DATA_CSV, holiday_csv=HOLIDAY_CSV, train_end=TRAIN_END, val_end=VAL_END,
                lookback=LOOKBACK, horizon=HORIZON):
    """Chronological train / validation / test windows; emissions z-scored with training statistics."""
    df = load_emissions(csv_path)
    tr_end, va_end = pd.to_datetime(train_end), pd.to_datetime(val_end)
    scalers, em = {}, pd.DataFrame(index=df.index)
    for c in SECTORS:
        tv = df.loc[:tr_end, c]
        scalers[c] = {'mean': tv.mean(), 'std': tv.std()}
        em[c] = (df[c] - scalers[c]['mean']) / (scalers[c]['std'] + 1e-8)
    cal = calendar_features(df, holiday_csv)

    def part(a, b):
        sl = slice(a, b)
        return windows(em.loc[sl].values, cal.loc[sl].values, df.loc[sl].index, lookback, horizon)

    lb = pd.Timedelta(days=lookback)
    return {'train': part(None, tr_end), 'val': part(tr_end - lb, va_end), 'test': part(va_end - lb, None),
            'scalers': scalers, 'raw_df': df}


def to_raw(y_scaled, scalers):
    """Invert the z-score scaling (B, H, S) -> Mt CO2/day."""
    out = np.empty_like(y_scaled, dtype=np.float64)
    for s, sec in enumerate(SECTORS):
        out[..., s] = y_scaled[..., s] * scalers[sec]['std'] + scalers[sec]['mean']
    return out
