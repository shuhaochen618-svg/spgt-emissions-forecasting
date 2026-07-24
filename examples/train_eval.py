import argparse
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import random

# Add repository root to python search path to run examples without full package installation
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from spgt import (
    TemporalSectoralTransformer,
    get_train_val_test_splits,
    SECTORS
)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def trend_regularized_loss(y_pred, y_true, lambda_trend, year_weights=None, device='cpu'):
    """
    Computes standard MSE loss plus an optional upward-trend penalty.

    The trend penalty is disabled by default and was not used for the
    manuscript benchmark table. It is kept only for sensitivity testing.
    """
    mse_loss = torch.mean((y_pred - y_true)**2)
    
    if lambda_trend == 0:
        return mse_loss, torch.tensor(0.0).to(device)
    
    # Optional sensitivity penalty: penalize upward trends in key sectors:
    # Industry (index 2) and Power (index 4).
    start_mean = y_pred[:, :5, [2, 4]].mean(dim=1)
    end_mean = y_pred[:, -5:, [2, 4]].mean(dim=1)
    trend = end_mean - start_mean
    
    trend_penalty = torch.mean(torch.clamp(trend, min=0.0)**2)
    
    if year_weights is not None:
        trend_penalty = torch.mean(year_weights * torch.clamp(trend, min=0.0)**2)
        
    total_loss = mse_loss + lambda_trend * trend_penalty
    return total_loss, trend_penalty

def train_one_epoch(model, loader, optimizer, lambda_trend, device):
    model.train()
    total_loss = 0
    total_penalty = 0
    for batch_x_hist, batch_x_fut, batch_y, batch_years in loader:
        batch_x_hist = batch_x_hist.float().to(device)
        batch_x_fut = batch_x_fut.float().to(device)
        batch_y = batch_y.float().to(device)
        batch_years = batch_years.float().to(device).unsqueeze(-1)
        
        # Year weights are only used if the optional trend penalty is enabled.
        year_weights = (batch_years - 2019.0) / 11.0
        
        optimizer.zero_grad()
        pred, _, _ = model(batch_x_hist, batch_x_fut)
        
        loss, penalty = trend_regularized_loss(pred, batch_y, lambda_trend, year_weights, device)
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
    parser.add_argument("--data_path", type=str, default="data/carbonmonitor-global_datas_2026-05-22.csv", help="Path to the Carbon Monitor CSV dataset.")
    parser.add_argument("--country", type=str, default="China", help="Target country name")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--lookback", type=int, default=90, help="Historical lookback window in days")
    parser.add_argument("--horizon", type=int, default=30, help="Forecast horizon in days")
    parser.add_argument("--lambda_trend", type=float, default=0.0, help="Optional upward-trend penalty weight; disabled by default.")
    args = parser.parse_args()

    set_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Prepare dataset path
    data_path = args.data_path
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset file not found: {data_path}. "
            "Please provide the Carbon Monitor CSV file with --data_path."
        )
        
    # 2. Preprocess data and split into Train, Validation, and Test sets
    print(f"Preprocessing data for {args.country} (lookback={args.lookback}, horizon={args.horizon})...")
    
    splits = get_train_val_test_splits(
        data_path, 
        country=args.country, 
        lookback=args.lookback, 
        horizon=args.horizon,
        train_end='2023-12-31', 
        val_end='2024-12-31'
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
    
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    
    # 3. Model initialization
    model = TemporalSectoralTransformer(
        lookback=args.lookback,
        horizon=args.horizon,
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
    print(f"\nTraining SPGT for {args.epochs} epochs (seed={args.seed}, lambda_trend={args.lambda_trend})...")
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        train_loss, train_pen = train_one_epoch(model, train_loader, optimizer, args.lambda_trend, device)
        val_mse, val_mae = evaluate(model, val_loader, device)
        scheduler.step()
        
        print(f"Epoch [{epoch+1:02d}/{args.epochs:02d}] "
              f"| Train Loss: {train_loss:.5f} (Trend Pen: {train_pen:.5f}) "
              f"| Val MSE (normalized): {val_mse:.5f} | Val MAE: {val_mae:.5f}")
        
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
    print(f"  Test MSE (normalized scale): {test_mse:.6f}")
    print(f"  Test MAE (normalized scale): {test_mae:.6f}")
    print("=========================================")

if __name__ == "__main__":
    main()
