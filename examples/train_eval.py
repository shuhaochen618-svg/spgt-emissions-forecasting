import argparse
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import tempfile

# Add repository root to python search path to run examples without full package installation
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from spgt import (
    TemporalSectoralTransformer,
    get_train_val_test_splits,
    SECTORS
)

# Helper function to generate realistic synthetic daily emission data
def generate_synthetic_dataset(csv_path):
    print("Generating a realistic synthetic daily emissions dataset...")
    # Define date range: 2021-01-01 to 2024-12-31 (4 years of daily data)
    dates = pd.date_range(start='2021-01-01', end='2024-12-31', freq='D')
    
    records = []
    for date in dates:
        # Base emission with some random noise and seasonal fluctuation
        day_of_year = date.dayofyear
        seasonality = 1.0 + 0.15 * np.cos(2 * np.pi * (day_of_year - 15) / 365.0) # low in winter/summer, high in spring/autumn
        
        # Shifting Lunar holiday dip: let's simulate Chinese New Year dip (usually late Jan or early Feb)
        # Let's say Feb 1 to Feb 15 has lower emissions
        is_holiday = (date.month == 2 and 1 <= date.day <= 15) or (date.month == 1 and 20 <= date.day <= 31)
        holiday_factor = 0.6 if is_holiday else 1.0
        
        for sector in SECTORS:
            # Different base scales for sectors
            if sector == 'Power':
                base = 15.0
            elif sector == 'Industry':
                base = 12.0
            elif sector == 'Ground Transport':
                base = 6.0
            elif sector == 'Residential':
                base = 3.0
            else:
                base = 1.0
                
            value = base * seasonality * holiday_factor + np.random.normal(0, base * 0.05)
            value = max(0.1, value) # keep it positive
            
            records.append({
                'date': date.strftime('%d/%m/%Y'),
                'country': 'China',
                'sector': sector,
                'value': value
            })
            
    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    print(f"Synthetic dataset saved to {csv_path}. Total samples: {len(df)}")

def policy_constrained_loss(y_pred, y_true, lambda_policy, year_weights=None, device='cpu'):
    """
    Computes standard MSE loss + optional policy constraint regularization.
    """
    mse_loss = torch.mean((y_pred - y_true)**2)
    
    if lambda_policy == 0:
        return mse_loss, torch.tensor(0.0).to(device)
    
    # Policy penalty: penalize upward trends in key sectors: Industry (index 2) & Power (index 4)
    start_mean = y_pred[:, :5, [2, 4]].mean(dim=1)
    end_mean = y_pred[:, -5:, [2, 4]].mean(dim=1)
    trend = end_mean - start_mean
    
    # Penalize upward trends (trend > 0)
    policy_penalty = torch.mean(torch.clamp(trend, min=0.0)**2)
    
    if year_weights is not None:
        policy_penalty = torch.mean(year_weights * torch.clamp(trend, min=0.0)**2)
        
    total_loss = mse_loss + lambda_policy * policy_penalty
    return total_loss, policy_penalty

def train_one_epoch(model, loader, optimizer, lambda_policy, device):
    model.train()
    total_loss = 0
    total_penalty = 0
    for batch_x_hist, batch_x_fut, batch_y, batch_years in loader:
        batch_x_hist = batch_x_hist.float().to(device)
        batch_x_fut = batch_x_fut.float().to(device)
        batch_y = batch_y.float().to(device)
        batch_years = batch_years.float().to(device).unsqueeze(-1)
        
        # Calculate policy constraint year weight (scale relative to target year 2030)
        year_weights = (batch_years - 2019.0) / 11.0
        
        optimizer.zero_grad()
        pred, _, _ = model(batch_x_hist, batch_x_fut)
        
        loss, penalty = policy_constrained_loss(pred, batch_y, lambda_policy, year_weights, device)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(batch_y)
        total_penalty += penalty.item() * len(batch_y)
        
    return total_loss / len(loader.dataset), total_penalty / len(loader.dataset)

def evaluate(model, loader, device):
    model.eval()
    total_mse = 0
    total_mae = 0
    with torch.no_grad():
        for batch_x_hist, batch_x_fut, batch_y, _ in loader:
            batch_x_hist = batch_x_hist.float().to(device)
            batch_x_fut = batch_x_fut.float().to(device)
            batch_y = batch_y.float().to(device)
            
            pred, _, _ = model(batch_x_hist, batch_x_fut)
            
            # Unregularized evaluation
            mse = torch.mean((pred - batch_y)**2)
            mae = torch.mean(torch.abs(pred - batch_y))
            
            total_mse += mse.item() * len(batch_y)
            total_mae += mae.item() * len(batch_y)
            
    return total_mse / len(loader.dataset), total_mae / len(loader.dataset)

def main():
    parser = argparse.ArgumentParser(description="SPGT Model Training and Evaluation Pipeline")
    parser.add_argument("--data_path", type=str, default="", help="Path to carbonmonitor CSV dataset. If omitted, runs in synthetic mode.")
    parser.add_argument("--country", type=str, default="China", help="Target country name")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--lambda_policy", type=float, default=0.1, help="Weight of policy constraint regularizer")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Prepare dataset path
    temp_csv_file = None
    if not args.data_path or not os.path.exists(args.data_path):
        print("Dataset file not specified or not found.")
        temp_csv = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
        temp_csv_file = temp_csv.name
        temp_csv.close()
        generate_synthetic_dataset(temp_csv_file)
        data_path = temp_csv_file
    else:
        data_path = args.data_path
        
    try:
        # 2. Preprocess data and split into Train, Val, Test sets
        lookback = 90
        horizon = 30
        print(f"Preprocessing data for {args.country} (lookback={lookback}, horizon={horizon})...")
        
        # Dates tailored for synthetic data or custom dataset
        splits = get_train_val_test_splits(
            data_path, 
            country=args.country, 
            lookback=lookback, 
            horizon=horizon,
            train_end='2023-12-31', 
            val_end='2024-06-30'
        )
        
        # Unpack splits
        X_train_hist, X_train_fut, y_train, train_dates = splits['train']
        X_val_hist, X_val_fut, y_val, val_dates = splits['val']
        X_test_hist, X_test_fut, y_test, test_dates = splits['test']
        
        train_years = np.array([d.year for d in train_dates])
        val_years = np.array([d.year for d in val_dates])
        test_years = np.array([d.year for d in test_dates])
        
        # PyTorch loaders
        train_ds = TensorDataset(torch.tensor(X_train_hist), torch.tensor(X_train_fut), torch.tensor(y_train), torch.tensor(train_years))
        val_ds = TensorDataset(torch.tensor(X_val_hist), torch.tensor(X_val_fut), torch.tensor(y_val), torch.tensor(val_years))
        test_ds = TensorDataset(torch.tensor(X_test_hist), torch.tensor(X_test_fut), torch.tensor(y_test), torch.tensor(test_years))
        
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
        
        # 3. Model initialization
        model = TemporalSectoralTransformer(
            lookback=lookback,
            horizon=horizon,
            num_sectors=len(SECTORS),
            num_calendar=5,
            d_model=32,
            nhead=4,
            num_layers=1,
            dropout=0.15
        ).to(device)
        
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        
        # 4. Training loop
        print(f"\nTraining SPGT for {args.epochs} epochs (lambda_policy={args.lambda_policy})...")
        best_val_loss = float('inf')
        
        for epoch in range(args.epochs):
            train_loss, train_pen = train_one_epoch(model, train_loader, optimizer, args.lambda_policy, device)
            val_mse, val_mae = evaluate(model, val_loader, device)
            scheduler.step()
            
            print(f"Epoch [{epoch+1:02d}/{args.epochs:02d}] "
                  f"| Train Loss: {train_loss:.5f} (Pen: {train_pen:.5f}) "
                  f"| Val MSE (Raw): {val_mse:.5f} | Val MAE: {val_mae:.5f}")
            
            if val_mse < best_val_loss:
                best_val_loss = val_mse
                torch.save(model.state_dict(), "spgt_best_weights.pt")
                
        # 5. Testing
        print("\nLoading best weights for final evaluation...")
        if os.path.exists("spgt_best_weights.pt"):
            model.load_state_dict(torch.load("spgt_best_weights.pt"))
            
        test_mse, test_mae = evaluate(model, test_loader, device)
        print("=========================================")
        print("             Final Test Metrics          ")
        print("=========================================")
        print(f"  Test MSE (Normalized scale): {test_mse:.6f}")
        print(f"  Test MAE (Normalized scale): {test_mae:.6f}")
        print("=========================================")
        
    finally:
        # Clean up temporary file
        if temp_csv_file and os.path.exists(temp_csv_file):
            os.remove(temp_csv_file)

if __name__ == "__main__":
    main()
