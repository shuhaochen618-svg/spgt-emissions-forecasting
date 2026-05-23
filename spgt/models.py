import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.20):
        super(AttentionLayer, self).__init__()
        self.mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.layernorm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.layernorm2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # Pre-LN Multihead Attention
        x_norm = self.layernorm(x)
        attn_out, attn_weights = self.mha(x_norm, x_norm, x_norm, need_weights=True)
        x = x + attn_out
        
        # Pre-LN FFN
        x_norm2 = self.layernorm2(x)
        ffn_out = self.ffn(x_norm2)
        x = x + ffn_out
        return x, attn_weights

class TemporalSectoralTransformer(nn.Module):
    """
    Spatiotemporal Patch Graph Transformer (SPGT / TSGT) for Carbon Emissions Forecasting.
    
    Combines:
    1. Channel-Independent Temporal Patching (similar to PatchTST)
    2. Shifting Lunar Calendar CNY Patch Embeddings
    3. Decoupled Cross-Sectoral Graph Attention with dynamic gating
    """
    def __init__(self, lookback=90, horizon=30, num_sectors=6, num_calendar=5, 
                 d_model=32, nhead=4, num_layers=1, dropout=0.15, patch_len=16, stride=8,
                 use_holiday=True, use_graph=True):
        super(TemporalSectoralTransformer, self).__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.num_sectors = num_sectors
        self.num_calendar = num_calendar
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.use_holiday = use_holiday
        self.use_graph = use_graph
        
        # Calculate padding and number of patches (pad lookback from 90 to 96)
        self.pad_len = 96 - lookback
        self.num_patches = (96 - patch_len) // stride + 1
        
        # Channel-independent patch projection
        self.patch_proj = nn.Linear(patch_len, d_model)
        
        # Future calendar patch projection (maps future calendar features to patch embeddings)
        self.cal_proj = nn.Linear(horizon * num_calendar, self.num_patches * d_model)
        
        # Gating projection for future calendar context to compute dynamic gate scalar
        self.gate_proj = nn.Linear(horizon * num_calendar, 1)
        
        # Positional and Sector Embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, 1, self.num_patches, d_model))
        self.sector_embed = nn.Parameter(torch.randn(1, num_sectors, 1, d_model))
        
        # Temporal Attention Layers (Shared across channels/sectors)
        self.temporal_layers = nn.ModuleList([
            AttentionLayer(d_model, nhead, dropout) for _ in range(num_layers)
        ])
        
        # Sectoral Attention Layers (Shared across patches)
        self.sector_layers = nn.ModuleList([
            AttentionLayer(d_model, nhead, dropout) for _ in range(num_layers)
        ])
        
        # Learnable gating parameter (initialized to 0.05 to start as almost pure PatchTST)
        self.gate = nn.Parameter(torch.tensor(0.05))
        
        # Output prediction head per sector (flat patch representation to horizon)
        self.head = nn.Linear(self.num_patches * d_model, horizon)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, X_hist, X_fut_cal):
        B, L, _ = X_hist.shape
        
        # Pad historical sequence to length 96 by repeating the first step
        if self.pad_len > 0:
            pad = X_hist[:, 0:1, :].repeat(1, self.pad_len, 1)
            X_hist_padded = torch.cat([pad, X_hist], dim=1) # (B, 96, S+C)
        else:
            X_hist_padded = X_hist
            
        emissions = X_hist_padded[:, :, :self.num_sectors] # (B, 96, S)
        
        # 1. Patching Emissions (Channel-Independence)
        emissions_t = emissions.transpose(1, 2).contiguous() # (B, S, 96)
        emissions_patches = emissions_t.unfold(2, self.patch_len, self.stride) # (B, S, N, P)
        em_proj = self.patch_proj(emissions_patches) # (B, S, N, d_model)
        
        # Combine representations and compute gate weight
        if self.use_holiday:
            # Project future calendar features (B, H, C) -> (B, N, d_model)
            flat_cal = X_fut_cal.reshape(B, -1)
            cal_proj = self.cal_proj(flat_cal).view(B, self.num_patches, self.d_model).unsqueeze(1) # (B, 1, N, d_model)
            x = em_proj + cal_proj + self.pos_embed + self.sector_embed
            
            # Compute dynamic gating weight from future calendar features
            # Force gate weight to be 0 if there is any active holiday in the future horizon (index 4 of calendar features)
            holiday_in_future = X_fut_cal[:, :, 4].max(dim=1, keepdim=True)[0]
            gate_weight = torch.sigmoid(self.gate_proj(flat_cal)) * (1.0 - holiday_in_future)
        else:
            x = em_proj + self.pos_embed + self.sector_embed
            gate_weight = torch.ones(B, 1).to(X_hist.device)
            
        x = self.dropout(x)
        
        # 3. Temporal Attention (Shared over sectors)
        x = x.view(B * self.num_sectors, self.num_patches, self.d_model)
        t_attn = None
        for layer in self.temporal_layers:
            x, t_attn = layer(x)
        x = x.view(B, self.num_sectors, self.num_patches, self.d_model)
        t_attn = t_attn.view(B, self.num_sectors, self.num_patches, self.num_patches)
        
        # 4. Decoupled Sector Graph Attention
        if self.use_graph:
            x_sector = x.permute(0, 2, 1, 3).contiguous().view(B * self.num_patches, self.num_sectors, self.d_model)
            x_sector_layer = x_sector.clone()
            s_attn = None
            for layer in self.sector_layers:
                x_sector_layer, s_attn = layer(x_sector_layer)
            
            # Dynamic gated connection: Z_final = Z_independent + gate * gw * Z_fused
            gw = gate_weight.view(B, 1, 1, 1)
            x_sector_reshaped = x_sector.view(B, self.num_patches, self.num_sectors, self.d_model)
            x_sector_layer_reshaped = x_sector_layer.view(B, self.num_patches, self.num_sectors, self.d_model)
            
            x_sector_updated = x_sector_reshaped + self.gate * gw * (x_sector_layer_reshaped - x_sector_reshaped)
            
            x = x_sector_updated.permute(0, 2, 1, 3).contiguous()
            s_attn_resized = s_attn.view(B, self.num_patches, self.num_sectors, self.num_sectors)
        else:
            s_attn_resized = torch.zeros(B, self.num_patches, self.num_sectors, self.num_sectors).to(x.device)
        
        # 5. Output Prediction Head
        x = x.view(B * self.num_sectors, -1) # (B * S, N * d_model)
        out = self.head(x) # (B * S, H)
        y_pred = out.view(B, self.num_sectors, self.horizon).transpose(1, 2).contiguous() # (B, H, S)
        
        # 7. Resize attention maps for compatibility with visualization script
        t_attn_resized = F.interpolate(t_attn, size=(self.lookback, self.lookback), mode='bilinear', align_corners=False)
        
        s_attn_temp = s_attn_resized.permute(0, 2, 3, 1).reshape(B * self.num_sectors * self.num_sectors, 1, self.num_patches)
        s_attn_temp = F.interpolate(s_attn_temp, size=self.lookback, mode='linear', align_corners=False)
        s_attn_resized = s_attn_temp.reshape(B, self.num_sectors, self.num_sectors, self.lookback).permute(0, 3, 1, 2).contiguous()
        
        return y_pred, t_attn_resized, s_attn_resized

class MovingAvg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on both ends
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        back = x[:, -1:, :].repeat(1, self.kernel_size // 2, 1)
        x = torch.cat([front, x, back], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class DLinearBaseline(nn.Module):
    def __init__(self, lookback=90, horizon=30, num_sectors=6, input_dim=11):
        super(DLinearBaseline, self).__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.num_sectors = num_sectors
        self.input_dim = input_dim
        
        self.decomp = SeriesDecomp(kernel_size=25)
        self.Linear_Seasonal = nn.Linear(lookback, horizon)
        self.Linear_Trend = nn.Linear(lookback, horizon)
        self.proj = nn.Linear(input_dim, num_sectors)
        
    def forward(self, X_hist, X_fut_cal):
        # X_hist: (B, L, input_dim)
        seasonal, trend = self.decomp(X_hist) # (B, L, input_dim)
        
        seasonal = seasonal.transpose(1, 2) # (B, input_dim, L)
        seasonal = self.Linear_Seasonal(seasonal) # (B, input_dim, H)
        seasonal = seasonal.transpose(1, 2) # (B, H, input_dim)
        
        trend = trend.transpose(1, 2) # (B, input_dim, L)
        trend = self.Linear_Trend(trend) # (B, input_dim, H)
        trend = trend.transpose(1, 2) # (B, H, input_dim)
        
        y_pred = seasonal + trend # (B, H, input_dim)
        y_pred = self.proj(y_pred) # (B, H, num_sectors)
        
        return y_pred, torch.zeros(1), torch.zeros(1)

class LSTMBaseline(nn.Module):
    def __init__(self, input_dim=11, hidden_dim=64, num_sectors=6, horizon=30):
        super(LSTMBaseline, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_sectors * horizon)
        self.num_sectors = num_sectors
        self.horizon = horizon
        
    def forward(self, X_hist, X_fut_cal):
        _, (h_n, _) = self.lstm(X_hist) # h_n: (1, B, hidden_dim)
        out = self.fc(h_n.squeeze(0)) # (B, num_sectors * horizon)
        y_pred = out.view(-1, self.horizon, self.num_sectors)
        return y_pred, torch.zeros(1), torch.zeros(1)

class GRUBaseline(nn.Module):
    def __init__(self, input_dim=11, hidden_dim=64, num_sectors=6, horizon=30):
        super(GRUBaseline, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_sectors * horizon)
        self.num_sectors = num_sectors
        self.horizon = horizon
        
    def forward(self, X_hist, X_fut_cal):
        _, h_n = self.gru(X_hist) # h_n: (1, B, hidden_dim)
        out = self.fc(h_n.squeeze(0)) # (B, num_sectors * horizon)
        y_pred = out.view(-1, self.horizon, self.num_sectors)
        return y_pred, torch.zeros(1), torch.zeros(1)

class RNNBaseline(nn.Module):
    def __init__(self, input_dim=11, hidden_dim=64, num_sectors=6, horizon=30):
        super(RNNBaseline, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_sectors * horizon)
        self.num_sectors = num_sectors
        self.horizon = horizon
        
    def forward(self, X_hist, X_fut_cal):
        _, h_n = self.rnn(X_hist) # h_n: (1, B, hidden_dim)
        out = self.fc(h_n.squeeze(0)) # (B, num_sectors * horizon)
        y_pred = out.view(-1, self.horizon, self.num_sectors)
        return y_pred, torch.zeros(1), torch.zeros(1)

class MLPBaseline(nn.Module):
    def __init__(self, input_dim=11, lookback=90, hidden_dim=128, num_sectors=6, horizon=30):
        super(MLPBaseline, self).__init__()
        self.lookback = lookback
        self.input_dim = input_dim
        self.horizon = horizon
        self.num_sectors = num_sectors
        
        self.fc = nn.Sequential(
            nn.Linear(lookback * input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_sectors * horizon)
        )
        
    def forward(self, X_hist, X_fut_cal):
        B = X_hist.shape[0]
        x = X_hist.view(B, -1)
        out = self.fc(x)
        y_pred = out.view(B, self.horizon, self.num_sectors)
        return y_pred, torch.zeros(1), torch.zeros(1)

class TransformerBaseline(nn.Module):
    def __init__(self, input_dim=11, num_sectors=6, d_model=64, nhead=4, num_layers=2, lookback=90, horizon=30, dropout=0.1):
        super(TransformerBaseline, self).__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.num_sectors = num_sectors
        
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, lookback, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.decoder_proj = nn.Linear(lookback * d_model, num_sectors * horizon)
        
    def forward(self, X_hist, X_fut_cal):
        B, L, _ = X_hist.shape
        x = self.input_proj(X_hist) # (B, L, d_model)
        x = x + self.pos_embed
        x = self.transformer_encoder(x) # (B, L, d_model)
        x = x.reshape(B, -1) # (B, L * d_model)
        out = self.decoder_proj(x) # (B, num_sectors * horizon)
        y_pred = out.view(B, self.horizon, self.num_sectors)
        return y_pred, torch.zeros(1), torch.zeros(1)

class PatchTSTBaseline(nn.Module):
    def __init__(self, input_dim=11, num_sectors=6, lookback=90, horizon=30, patch_len=16, stride=8, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(PatchTSTBaseline, self).__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.patch_len = patch_len
        self.stride = stride
        self.input_dim = input_dim
        self.num_sectors = num_sectors
        
        self.pad_len = 6 # 90 + 6 = 96. (96 - 16)/8 + 1 = 11 patches.
        self.num_patches = (lookback + self.pad_len - patch_len) // stride + 1
        
        self.patch_proj = nn.Linear(patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.head = nn.Linear(self.num_patches * d_model, horizon)
        
    def forward(self, X_hist, X_fut_cal):
        B, L, M = X_hist.shape
        
        # Channel Independence (CI)
        x = X_hist.transpose(1, 2).reshape(B * M, L) # (B * M, L)
        
        if self.pad_len > 0:
            x = F.pad(x, (self.pad_len, 0), mode='replicate') # (B * M, L + pad_len)
            
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride) # (B * M, num_patches, patch_len)
        x = self.patch_proj(x) # (B * M, num_patches, d_model)
        x = x + self.pos_embed
        x = self.transformer_encoder(x) # (B * M, num_patches, d_model)
        x = x.reshape(B * M, -1) # (B * M, num_patches * d_model)
        x = self.head(x) # (B * M, H)
        
        y_pred = x.view(B, M, self.horizon).transpose(1, 2) # (B, H, M)
        y_pred = y_pred[:, :, :self.num_sectors] # (B, H, num_sectors)
        return y_pred, torch.zeros(1), torch.zeros(1)

if __name__ == "__main__":
    # Dry run
    B, L, H = 16, 90, 30
    model = TemporalSectoralTransformer(lookback=L, horizon=H)
    X_hist = torch.randn(B, L, 6 + 5)
    X_fut_cal = torch.randn(B, H, 5)
    y_pred, t_attn, s_attn = model(X_hist, X_fut_cal)
    print("Output shape:", y_pred.shape) # (16, 30, 6)
    print("Temporal Attn shape:", t_attn.shape) # (16, 6, 90, 90)
    print("Sector Attn shape:", s_attn.shape) # (16, 90, 6, 6)
    assert y_pred.shape == (B, H, 6)
    assert t_attn.shape == (B, 6, L, L)
    assert s_attn.shape == (B, L, 6, 6)
    print("Model Dry Run SUCCESSFUL!")
