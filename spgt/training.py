import random

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loaders(splits, batch_size=64, eval_batch_size=128):
    def ds(part):
        Xh, Xf, y, _ = splits[part]
        return TensorDataset(torch.tensor(Xh), torch.tensor(Xf), torch.tensor(y))
    return (DataLoader(ds('train'), batch_size=batch_size, shuffle=True),
            DataLoader(ds('val'), batch_size=eval_batch_size, shuffle=False),
            DataLoader(ds('test'), batch_size=eval_batch_size, shuffle=False))


def train_model(model, train_loader, val_loader, epochs, lr, weight_decay, patience=20, verbose=False):
    """AdamW + cosine annealing on MSE; returns the weights with the lowest validation MSE."""
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best, best_w, bad = float('inf'), None, 0
    for ep in range(epochs):
        model.train()
        for xh, xf, y in train_loader:
            xh, xf, y = xh.float().to(device), xf.float().to(device), y.float().to(device)
            opt.zero_grad()
            loss = torch.mean((model(xh, xf)[0] - y) ** 2)
            loss.backward()
            opt.step()
        model.eval()
        tot = 0.0
        with torch.no_grad():
            for xh, xf, y in val_loader:
                xh, xf, y = xh.float().to(device), xf.float().to(device), y.float().to(device)
                tot += torch.mean((model(xh, xf)[0] - y) ** 2).item() * len(y)
        vl = tot / len(val_loader.dataset)
        sch.step()
        if verbose:
            print(f'epoch {ep + 1:3d}  val_mse {vl:.5f}')
        if vl < best:
            best, best_w, bad = vl, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        if bad >= patience:
            break
    if best_w is not None:
        model.load_state_dict(best_w)
    return model, best, ep + 1


def predict(model, loader):
    """Scaled predictions (N, H, S)."""
    model.eval()
    preds = []
    with torch.no_grad():
        for xh, xf, _ in loader:
            preds.append(model(xh.float().to(device), xf.float().to(device))[0].cpu().numpy())
    return np.concatenate(preds, axis=0)
