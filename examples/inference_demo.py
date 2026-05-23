import sys
import os
import torch

# Add repository root to python search path to run examples without full package installation
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from spgt import TemporalSectoralTransformer

def main():
    print("Initializing Spatiotemporal Patch Graph Transformer (SPGT)...")
    
    # Define shapes matching the paper layout
    batch_size = 4
    lookback = 90
    horizon = 30
    num_sectors = 6
    num_calendar = 5
    
    # Hist input: emissions (6 sectors) + calendar features (5 features) = 11 channels
    input_dim = num_sectors + num_calendar
    
    # Initialize the model
    model = TemporalSectoralTransformer(
        lookback=lookback, 
        horizon=horizon, 
        num_sectors=num_sectors, 
        num_calendar=num_calendar,
        d_model=32,
        nhead=4,
        num_layers=2,
        dropout=0.1
    )
    
    # Generate dummy input tensors
    X_hist = torch.randn(batch_size, lookback, input_dim)
    X_fut_cal = torch.randn(batch_size, horizon, num_calendar)
    
    print(f"Input Shapes:")
    print(f"  Historical features (X_hist):    {X_hist.shape}")
    print(f"  Future calendar features (X_fut): {X_fut_cal.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        y_pred, t_attn, s_attn = model(X_hist, X_fut_cal)
        
    print("\nForward pass successful!")
    print(f"Output Shapes:")
    print(f"  Emissions Predictions (y_pred):   {y_pred.shape} (Expected: {batch_size}, {horizon}, {num_sectors})")
    print(f"  Temporal Attention maps (t_attn): {t_attn.shape} (Expected: {batch_size}, {num_sectors}, {lookback}, {lookback})")
    print(f"  Sectoral Attention maps (s_attn): {s_attn.shape} (Expected: {batch_size}, {lookback}, {num_sectors}, {num_sectors})")

if __name__ == "__main__":
    main()
