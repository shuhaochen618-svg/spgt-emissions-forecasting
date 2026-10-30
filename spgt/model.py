import torch
import torch.nn as nn


class AttentionLayer(nn.Module):
    """Pre-LN multi-head self-attention block with a feed-forward sublayer."""

    def __init__(self, d_model, nhead, dropout=0.2):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.layernorm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.layernorm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        h = self.layernorm(x)
        attn_out, attn_weights = self.mha(h, h, h, need_weights=True)
        x = x + attn_out
        x = x + self.ffn(self.layernorm2(x))
        return x, attn_weights


class TemporalSectoralTransformer(nn.Module):
    """SPGT: Sectoral Patch Gated Transformer.

    Inputs:  X_hist (B, L, S + C) scaled emissions and calendar features of the lookback window,
             X_fut_cal (B, H, C) calendar features of the forecast horizon.
    Output:  y_pred (B, H, S), temporal attention (B, S, N, N), sectoral attention (B, N, S, S).
    Calendar column 4 must be the CNY-window indicator.
    """

    def __init__(self, lookback=90, horizon=30, num_sectors=6, num_calendar=5,
                 d_model=32, nhead=4, num_layers=1, dropout=0.15, patch_len=16, stride=8,
                 use_holiday=True, use_graph=True):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.num_sectors = num_sectors
        self.num_calendar = num_calendar
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.use_holiday = use_holiday
        self.use_graph = use_graph

        # lookback is left-padded to 96 steps
        self.pad_len = 96 - lookback
        self.num_patches = (96 - patch_len) // stride + 1

        self.patch_proj = nn.Linear(patch_len, d_model)
        self.cal_proj = nn.Linear(horizon * num_calendar, self.num_patches * d_model)
        self.gate_proj = nn.Linear(horizon * num_calendar, 1)
        self.pos_embed = nn.Parameter(torch.randn(1, 1, self.num_patches, d_model))
        self.sector_embed = nn.Parameter(torch.randn(1, num_sectors, 1, d_model))
        self.temporal_layers = nn.ModuleList([AttentionLayer(d_model, nhead, dropout) for _ in range(num_layers)])
        self.sector_layers = nn.ModuleList([AttentionLayer(d_model, nhead, dropout) for _ in range(num_layers)])
        self.gate = nn.Parameter(torch.tensor(0.05))
        self.head = nn.Linear(self.num_patches * d_model, horizon)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X_hist, X_fut_cal):
        B = X_hist.shape[0]
        S, N, D = self.num_sectors, self.num_patches, self.d_model

        if self.pad_len > 0:
            pad = X_hist[:, 0:1, :].repeat(1, self.pad_len, 1)
            X_hist = torch.cat([pad, X_hist], dim=1)  # (B, 96, S + C)

        emissions = X_hist[:, :, :S].transpose(1, 2).contiguous()  # (B, S, 96)
        patches = emissions.unfold(2, self.patch_len, self.stride)  # (B, S, N, P)
        em_proj = self.patch_proj(patches)  # (B, S, N, D)

        if self.use_holiday:
            flat_cal = X_fut_cal.reshape(B, -1)
            cal_proj = self.cal_proj(flat_cal).view(B, N, D).unsqueeze(1)  # (B, 1, N, D)
            x = em_proj + cal_proj + self.pos_embed + self.sector_embed
            # gate is zero when any horizon day lies in the CNY window
            holiday_in_future = X_fut_cal[:, :, 4].max(dim=1, keepdim=True)[0]
            gate_weight = torch.sigmoid(self.gate_proj(flat_cal)) * (1.0 - holiday_in_future)
        else:
            x = em_proj + self.pos_embed + self.sector_embed
            gate_weight = torch.ones(B, 1).to(X_hist.device)

        x = self.dropout(x)

        x = x.view(B * S, N, D)
        t_attn = None
        for layer in self.temporal_layers:
            x, t_attn = layer(x)
        x = x.view(B, S, N, D)
        t_attn = t_attn.view(B, S, N, N)

        if self.use_graph:
            x_sector = x.permute(0, 2, 1, 3).contiguous().view(B * N, S, D)
            z = x_sector.clone()
            s_attn = None
            for layer in self.sector_layers:
                z, s_attn = layer(z)
            gw = gate_weight.view(B, 1, 1, 1)
            x_ind = x_sector.view(B, N, S, D)
            x_fused = z.view(B, N, S, D)
            x = (x_ind + self.gate * gw * (x_fused - x_ind)).permute(0, 2, 1, 3).contiguous()
            s_attn = s_attn.view(B, N, S, S)
        else:
            s_attn = torch.zeros(B, N, S, S).to(x.device)

        out = self.head(x.view(B * S, -1))  # (B * S, H)
        y_pred = out.view(B, S, self.horizon).transpose(1, 2).contiguous()  # (B, H, S)
        return y_pred, t_attn, s_attn


SPGT = TemporalSectoralTransformer


def build_spgt(cfg, lookback, horizon, num_calendar):
    """Build SPGT from a model config dict and set the initial gate value."""
    model = TemporalSectoralTransformer(
        lookback, horizon, num_calendar=num_calendar, d_model=cfg['d_model'], nhead=cfg['nhead'],
        num_layers=cfg['num_layers'], dropout=cfg['dropout'], patch_len=cfg['patch_len'], stride=cfg['stride'],
        use_holiday=True, use_graph=cfg['use_graph'])
    with torch.no_grad():
        model.gate.fill_(cfg['gate_init'])
    return model
