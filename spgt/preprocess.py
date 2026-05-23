import pandas as pd
import numpy as np

# Shifting cultural holidays
CNY_DATES = {
    2018: '2018-02-16', 2019: '2019-02-05', 2020: '2020-01-25', 2021: '2021-02-12',
    2022: '2022-02-01', 2023: '2023-01-22', 2024: '2024-02-10', 2025: '2025-01-29',
    2026: '2026-02-17', 2027: '2027-02-06', 2028: '2028-01-26', 2029: '2029-02-13',
    2030: '2030-02-03', 2031: '2031-01-23'
}

DIWALI_DATES = {
    2018: '2018-11-07', 2019: '2019-10-27', 2020: '2020-11-14', 2021: '2021-11-04',
    2022: '2022-10-24', 2023: '2023-11-12', 2024: '2024-10-31', 2025: '2025-10-20',
    2026: '2026-11-08', 2027: '2027-10-29', 2028: '2028-10-17', 2029: '2029-11-05',
    2030: '2030-10-26', 2031: '2031-11-14'
}

SECTORS = ['Domestic Aviation', 'Ground Transport', 'Industry', 'International Aviation', 'Power', 'Residential']
CALENDAR_COLS = ['month', 'day_of_week', 'day_of_year', 'holiday_dist', 'holiday_active']

def get_holiday_distance(date, country):
    """
    Calculate country-specific cultural holiday distances:
    - China: Chinese New Year (shifting)
    - India: Diwali (shifting)
    - United States / EU27: Christmas (Dec 25, fixed)
    - Japan: New Year / Shogatsu (Jan 1, fixed)
    """
    dt = pd.to_datetime(date)
    year = dt.year
    
    if country == 'China':
        dates_dict = CNY_DATES
    elif country == 'India':
        dates_dict = DIWALI_DATES
    elif country in ['United States', 'EU27']:
        # Fixed Christmas (Dec 25)
        closest_cny = pd.to_datetime(f"{year}-12-25")
        # Check adjacent years
        options = [pd.to_datetime(f"{y}-12-25") for y in [year-1, year, year+1]]
        closest_cny = min(options, key=lambda d: abs((dt - d).days))
        return (dt - closest_cny).days
    elif country == 'Japan':
        # Fixed New Year (Jan 1)
        options = [pd.to_datetime(f"{y}-01-01") for y in [year-1, year, year+1]]
        closest_cny = min(options, key=lambda d: abs((dt - d).days))
        return (dt - closest_cny).days
    else:
        # Default fallback to Jan 1
        options = [pd.to_datetime(f"{y}-01-01") for y in [year-1, year, year+1]]
        closest_cny = min(options, key=lambda d: abs((dt - d).days))
        return (dt - closest_cny).days
        
    # For shifting holidays (China and India)
    options = []
    for y in [year - 1, year, year + 1]:
        if y in dates_dict:
            options.append(pd.to_datetime(dates_dict[y]))
            
    closest_h = min(options, key=lambda d: abs((dt - d).days))
    return (dt - closest_h).days

def load_and_preprocess(csv_path, country='China'):
    """
    Load raw Carbon Monitor data and extract calendar and holiday features.
    """
    df = pd.read_csv(csv_path)
    
    # Filter country
    df_country = df[df['country'] == country].copy()
    # Try different date formats for robustness
    try:
        df_country['date_parsed'] = pd.to_datetime(df_country['date'], format='%d/%m/%Y')
    except Exception:
        df_country['date_parsed'] = pd.to_datetime(df_country['date'])
        
    df_country = df_country.sort_values('date_parsed')
    
    # Pivot sectors
    df_pivot = df_country.pivot(index='date_parsed', columns='sector', values='value')
    df_pivot = df_pivot.ffill().bfill()
    
    # Calculate calendar features
    df_pivot['month'] = df_pivot.index.month
    df_pivot['day_of_week'] = df_pivot.index.dayofweek
    df_pivot['day_of_year'] = df_pivot.index.dayofyear
    
    # Calculate country-specific holiday distance
    df_pivot['holiday_dist'] = [get_holiday_distance(d, country) for d in df_pivot.index]
    
    # Active window indicator:
    # China/India: -7 to +15 days. US/EU: -5 to +5 days. Japan: -3 to +5 days.
    if country in ['China', 'India']:
        df_pivot['holiday_active'] = ((df_pivot['holiday_dist'] >= -7) & (df_pivot['holiday_dist'] <= 15)).astype(float)
    elif country in ['United States', 'EU27']:
        df_pivot['holiday_active'] = ((df_pivot['holiday_dist'] >= -5) & (df_pivot['holiday_dist'] <= 5)).astype(float)
    else:
        df_pivot['holiday_active'] = ((df_pivot['holiday_dist'] >= -3) & (df_pivot['holiday_dist'] <= 5)).astype(float)
        
    return df_pivot

def scale_data(df, train_idx_end):
    """
    Normalize sector emissions and scale calendar values.
    """
    df_scaled = df.copy()
    scalers = {}
    for col in SECTORS:
        train_vals = df.loc[:train_idx_end, col]
        mean = train_vals.mean()
        std = train_vals.std()
        df_scaled[col] = (df[col] - mean) / (std + 1e-8)
        scalers[col] = {'mean': mean, 'std': std}
        
    df_scaled['month'] = df['month'] / 12.0
    df_scaled['day_of_week'] = df['day_of_week'] / 7.0
    df_scaled['day_of_year'] = df['day_of_year'] / 365.0
    df_scaled['holiday_dist'] = df['holiday_dist'] / 180.0
    return df_scaled, scalers

def create_sliding_windows(df, lookback, horizon):
    """
    Generate historical sliding windows and future calendar lookaheads.
    """
    emissions = df[SECTORS].values
    calendar = df[CALENDAR_COLS].values
    dates = df.index
    
    X_hist, X_fut_cal, y, sample_dates = [], [], [], []
    num_days = len(df)
    
    for i in range(num_days - lookback - horizon + 1):
        hist_emissions = emissions[i : i + lookback]
        hist_calendar = calendar[i : i + lookback]
        hist_feat = np.hstack([hist_emissions, hist_calendar])
        
        fut_calendar = calendar[i + lookback : i + lookback + horizon]
        fut_emissions = emissions[i + lookback : i + lookback + horizon]
        
        X_hist.append(hist_feat)
        X_fut_cal.append(fut_calendar)
        y.append(fut_emissions)
        sample_dates.append(dates[i + lookback])
        
    return np.array(X_hist), np.array(X_fut_cal), np.array(y), sample_dates

def get_train_val_test_splits(csv_path, country='China', lookback=90, horizon=30, train_end='2023-12-31', val_end='2024-12-31'):
    """
    Load data and partition into Train, Validation, and Test sets based on dates.
    """
    df = load_and_preprocess(csv_path, country)
    
    train_end_date = pd.to_datetime(train_end)
    val_end_date = pd.to_datetime(val_end)
    
    df_scaled, scalers = scale_data(df, train_end_date)
    
    train_df = df_scaled.loc[:train_end_date]
    val_df = df_scaled.loc[train_end_date - pd.Timedelta(days=lookback): val_end_date]
    test_df = df_scaled.loc[val_end_date - pd.Timedelta(days=lookback):]
    
    X_train_hist, X_train_fut, y_train, train_dates = create_sliding_windows(train_df, lookback, horizon)
    X_val_hist, X_val_fut, y_val, val_dates = create_sliding_windows(val_df, lookback, horizon)
    X_test_hist, X_test_fut, y_test, test_dates = create_sliding_windows(test_df, lookback, horizon)
    
    return {
        'train': (X_train_hist, X_train_fut, y_train, train_dates),
        'val': (X_val_hist, X_val_fut, y_val, val_dates),
        'test': (X_test_hist, X_test_fut, y_test, test_dates),
        'scalers': scalers,
        'raw_df': df
    }

if __name__ == "__main__":
    print("Preprocess module loaded. Running verification...")
    # Test on a dummy DataFrame
    date_range = pd.date_range(start='2023-01-01', end='2023-03-31')
    dummy_data = pd.DataFrame(index=date_range)
    dummy_data['country'] = 'China'
    dummy_data['date'] = date_range.strftime('%d/%m/%Y')
    for sector in SECTORS:
        dummy_data[sector] = np.random.uniform(5.0, 15.0, size=len(date_range))
    
    # Flatten sectors to match Carbon Monitor long format
    long_format_list = []
    for sector in SECTORS:
        sec_df = pd.DataFrame({
            'date': dummy_data['date'],
            'country': 'China',
            'sector': sector,
            'value': dummy_data[sector]
        })
        long_format_list.append(sec_df)
    
    long_df = pd.concat(long_format_list)
    
    # Save temporarily to a dummy file to test load_and_preprocess
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        temp_csv = f.name
    try:
        long_df.to_csv(temp_csv, index=False)
        processed = load_and_preprocess(temp_csv, country='China')
        print(f"Preprocessed DataFrame columns: {list(processed.columns)}")
        print(f"Processed shape: {processed.shape}")
        
        # Test splits
        splits = get_train_val_test_splits(temp_csv, country='China', lookback=10, horizon=5, train_end='2023-02-15', val_end='2023-03-10')
        print("Dataset splitting successful!")
        for name, split in [('Train', splits['train']), ('Val', splits['val']), ('Test', splits['test'])]:
            print(f"  {name} shapes: Hist={split[0].shape}, Fut={split[1].shape}, Target={split[2].shape}")
    finally:
        if os.path.exists(temp_csv):
            os.remove(temp_csv)
    print("Verification complete.")
