"""
EEGNet 训练脚本 — 适配本仓库 EEG_Test 流程

数据：由 `data_process/data_process.ipynb` 第 8 节导出的 NPZ
  路径默认：`data_process/out/*_mi.npz`
  键：`X` (N, C, T) 已按 trial z-score；`y` 为 0=左、1=右（与 TRIG 1/2 对应）
  通道：8 EEG（PO7, PO3, POz, PO4, PO8, O1, Oz, O2）

运行（在项目根或 Classification 目录下均可）：
  python Classification/eegnet.py
  python Classification/eegnet.py --npz path/to/file_mi.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

# ======================
# 0) 工具
# ======================
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def accuracy_from_logits(logits, y):
    pred = torch.argmax(logits, dim=1)
    return (pred == y).float().mean().item()

def run_one_epoch(model, loader, optimizer, criterion, device, train=True):
    model.train(train)
    total_loss, total_acc, total_n = 0.0, 0.0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_acc  += accuracy_from_logits(logits, y) * bs
        total_n    += bs

    return total_loss / total_n, total_acc / total_n


# ======================
# 1) Dataset
# ======================
class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()  # (N,C,T)
        self.y = torch.from_numpy(y).long()
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        x = self.X[i].unsqueeze(0)  # -> (1,C,T)
        return x, self.y[i]


# ======================
# 2) EEGNet (简化版)
# ======================
class EEGNet(nn.Module):
    """
    输入:  (B,1,C,T)
    输出:  (B,n_classes)

    参考 EEGNet: temporal conv -> depthwise spatial conv -> separable conv -> FC
    """
    def __init__(self, n_chans, n_times, n_classes,
                 F1=8, D=2, F2=16,
                 kern_length=32,
                 dropout=0.5):
        super().__init__()
        # Block 1: Temporal Conv
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=F1,
            kernel_size=(1, kern_length),
            padding=(0, kern_length // 2),
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(F1)

        # Block 1: Depthwise Spatial Conv (across channels)
        self.depthwise = nn.Conv2d(
            in_channels=F1, out_channels=F1 * D,
            kernel_size=(n_chans, 1),
            groups=F1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.act = nn.ELU()
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))
        self.drop1 = nn.Dropout(dropout)

        # Block 2: Separable Conv (depthwise temporal + pointwise)
        self.sep_depth = nn.Conv2d(
            in_channels=F1 * D, out_channels=F1 * D,
            kernel_size=(1, 16),
            padding=(0, 16 // 2),
            groups=F1 * D,
            bias=False
        )
        self.sep_point = nn.Conv2d(
            in_channels=F1 * D, out_channels=F2,
            kernel_size=(1, 1),
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8))
        self.drop2 = nn.Dropout(dropout)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_chans, n_times)
            h = self._forward_features(dummy)
            feat_dim = h.shape[1]

        self.classifier = nn.Linear(feat_dim, n_classes)

    def _forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)

        x = self.depthwise(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.sep_depth(x)
        x = self.sep_point(x)
        x = self.bn3(x)
        x = self.act(x)
        x = self.pool2(x)
        x = self.drop2(x)

        x = x.flatten(1)
        return x

    def forward(self, x):
        feat = self._forward_features(x)
        return self.classifier(feat)


# ======================
# 3) 单折训练（early stopping）
# ======================
def train_one_fold(X_train, y_train, X_val, y_val, X_test, y_test,
                   n_chans, n_times, n_classes,
                   batch_size=32, epochs=200,
                   lr=1e-3, weight_decay=1e-4,
                   patience=20, warmup_epochs=10,
                   seed=42,
                   kern_length=64,
                   dropout=0.25):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    train_loader = DataLoader(EEGDataset(X_train, y_train), batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(EEGDataset(X_val, y_val),     batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(EEGDataset(X_test, y_test),   batch_size=batch_size, shuffle=False)

    model = EEGNet(
        n_chans=n_chans, n_times=n_times, n_classes=n_classes,
        F1=8, D=2, F2=16,
        kern_length=kern_length, dropout=dropout
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=6)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_state = None
    bad = 0

    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = run_one_epoch(model, train_loader, optimizer, criterion, device, train=True)
        va_loss, va_acc = run_one_epoch(model, val_loader,   optimizer, criterion, device, train=False)
        sched.step(va_acc)

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            if ep > warmup_epochs:
                bad += 1
                if bad >= patience:
                    break

    model.load_state_dict(best_state)
    te_loss, te_acc = run_one_epoch(model, test_loader, optimizer, criterion, device, train=False)
    return best_val_acc, te_acc, best_state


def _effective_stratified_k(y: np.ndarray, k: int) -> int:
    """分层 K 折：每类至少 k 个样本才能做 k 折；否则降低折数。"""
    uniq, counts = np.unique(y, return_counts=True)
    min_count = int(counts.min())
    return max(2, min(k, min_count, len(y) // 2))


# ======================
# 4) K 折交叉验证
# ======================
def kfold_eegnet(X, y, k=10, inner_val_ratio=0.15, seed=42,
                 batch_size=32, epochs=200, lr=1e-3, weight_decay=1e-4,
                 patience=20, warmup_epochs=10,
                 save_dir="./results", save_best=True,
                 kern_length=64,
                 dropout=0.25):
    set_seed(seed)
    os.makedirs(save_dir, exist_ok=True)

    u = np.unique(y)
    mapping = {lab: i for i, lab in enumerate(u)}
    y = np.vectorize(mapping.get)(y).astype(int)

    n_chans = X.shape[1]
    n_times = X.shape[2]
    n_classes = len(np.unique(y))

    k_eff = _effective_stratified_k(y, k)
    if k_eff < k:
        print(f"[warn] 样本/类别数限制：StratifiedKFold 从 k={k} 降为 k_eff={k_eff}")

    skf = StratifiedKFold(n_splits=k_eff, shuffle=True, random_state=seed)

    test_accs = []
    val_accs = []
    fold_states = []

    for fold, (trainval_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_trainval, y_trainval = X[trainval_idx], y[trainval_idx]
        X_test, y_test         = X[test_idx], y[test_idx]

        sss = StratifiedShuffleSplit(n_splits=1, test_size=inner_val_ratio, random_state=seed + fold)
        tr_idx, va_idx = next(sss.split(X_trainval, y_trainval))
        X_train, y_train = X_trainval[tr_idx], y_trainval[tr_idx]
        X_val,   y_val   = X_trainval[va_idx], y_trainval[va_idx]

        best_val_acc, test_acc, best_state = train_one_fold(
            X_train, y_train, X_val, y_val, X_test, y_test,
            n_chans=n_chans, n_times=n_times, n_classes=n_classes,
            batch_size=batch_size, epochs=epochs,
            lr=lr, weight_decay=weight_decay,
            patience=patience, warmup_epochs=warmup_epochs,
            seed=seed + fold,
            kern_length=kern_length,
            dropout=dropout,
        )

        val_accs.append(best_val_acc)
        test_accs.append(test_acc)
        fold_states.append(best_state)
        print(f"[Fold {fold}/{k_eff}] best val acc={best_val_acc:.4f} | test acc={test_acc:.4f}")

    test_accs = np.array(test_accs, dtype=float)
    val_accs = np.array(val_accs, dtype=float)

    best_fold_idx = int(np.argmax(test_accs))
    best_fold = best_fold_idx + 1
    best_test_acc = float(test_accs[best_fold_idx])
    best_val_acc = float(val_accs[best_fold_idx])

    print("\n=== EEGNet K-Fold Summary ===")
    print("Test acc per fold:", np.round(test_accs, 4).tolist())
    print(f"Mean ± Std = {test_accs.mean():.4f} ± {test_accs.std(ddof=1):.4f}")
    print(f"Best fold: {best_fold} (test_acc={best_test_acc:.4f})")

    best_params_dict = {
        "best_fold": best_fold,
        "best_val_acc": best_val_acc,
        "best_test_acc": best_test_acc,
        "test_accs_per_fold": test_accs.tolist(),
        "val_accs_per_fold": val_accs.tolist(),
        "mean_test_acc": float(test_accs.mean()),
        "std_test_acc": float(test_accs.std(ddof=1)),
        "hyperparams": {
            "k_requested": k,
            "k_effective": k_eff,
            "inner_val_ratio": inner_val_ratio,
            "seed": seed,
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "patience": patience,
            "warmup_epochs": warmup_epochs,
            "n_chans": n_chans,
            "n_times": n_times,
            "n_classes": n_classes,
            "kern_length": kern_length,
            "dropout": dropout,
        },
    }

    if save_best:
        best_state = fold_states[best_fold_idx]
        model_path = os.path.join(save_dir, "eegnet_best.pt")
        torch.save(best_state, model_path)
        print(f"Saved best model state -> {model_path}")

        params_path = os.path.join(save_dir, "best_params.json")
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(best_params_dict, f, indent=2, ensure_ascii=False)
        print(f"Saved best params dict -> {params_path}")

    return test_accs, best_params_dict


# ======================
# 5) 与本项目 NPZ 对齐的加载
# ======================
EEG_CH_NAMES_EXPECTED = ("PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2")


def project_root() -> Path:
    """Classification/eegnet.py -> 项目根 EEG_Test。"""
    return Path(__file__).resolve().parent.parent


def default_npz_path() -> Path | None:
    """`data_process/out/*_mi.npz` 中最新修改的一个；若无则 None。"""
    out_dir = project_root() / "data_process" / "out"
    if not out_dir.is_dir():
        return None
    candidates = sorted(out_dir.glob("*_mi.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def load_mi_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    读取 `data_process.ipynb` 导出的 NPZ。
    返回 X (float32), y (int64), meta 字典（sfreq、ch_names、times 等）。
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"NPZ 不存在: {path}")

    d = np.load(path, allow_pickle=True)
    if "X" not in d or "y" not in d:
        raise KeyError(f"NPZ 需包含键 'X' 与 'y'：{path}")

    X = np.asarray(d["X"], dtype=np.float32)
    y = np.asarray(d["y"], dtype=np.int64)

    if X.ndim != 3:
        raise ValueError(f"X 应为 (N,C,T)，当前 shape={X.shape}")
    n_ch = X.shape[1]
    if n_ch != 8:
        print(f"[warn] 本仓库 FIF 为 8 路 EEG，当前 X 通道数={n_ch}（若仍为多分类可忽略）")

    meta = {
        "path": str(path.resolve()),
        "sfreq": float(d["sfreq"]) if "sfreq" in d else None,
        "ch_names": d["ch_names"] if "ch_names" in d else None,
        "times": np.asarray(d["times"]) if "times" in d else None,
        "tmin": float(d["tmin"]) if "tmin" in d else None,
        "tmax": float(d["tmax"]) if "tmax" in d else None,
    }
    if meta["ch_names"] is not None:
        names = [str(x) for x in meta["ch_names"].tolist()]
        if tuple(names) != EEG_CH_NAMES_EXPECTED:
            print(f"[warn] 通道名与采集顺序不一致: {names[:3]}... 期望 {EEG_CH_NAMES_EXPECTED}")

    return X, y, meta


def _parse_args():
    p = argparse.ArgumentParser(description="EEGNet 训练（EEG_Test 八通道 MI NPZ）")
    p.add_argument(
        "--npz",
        type=str,
        default=None,
        help="NPZ 路径；默认取 data_process/out 下最新 *_mi.npz",
    )
    p.add_argument("--k", type=int, default=10, help="K 折（会按每类样本数自动降折）")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="结果目录；默认 Classification/results",
    )
    p.add_argument("--kern-length", type=int, default=64, help="时间卷积核长度（与采样率 500Hz 下 ~128ms 量级）")
    return p.parse_args()


# ======================
# 6) 运行
# ======================
if __name__ == "__main__":
    args = _parse_args()

    npz_path = Path(args.npz).resolve() if args.npz else default_npz_path()
    if npz_path is None or not npz_path.is_file():
        print(
            "未找到 NPZ：请先运行 data_process/data_process.ipynb 第 8 节导出，\n"
            "或使用: python Classification/eegnet.py --npz data_process/out/xxx_mi.npz",
            file=sys.stderr,
        )
        sys.exit(1)

    X, y, meta = load_mi_npz(npz_path)
    print("NPZ:", meta["path"])
    print("X:", X.shape, X.dtype)
    print("y:", y.shape, y.dtype, "classes:", np.unique(y, return_counts=True))
    if meta["sfreq"] is not None:
        print("sfreq:", meta["sfreq"], "Hz")

    save_dir = args.save_dir or str(Path(__file__).resolve().parent / "results")

    kfold_eegnet(
        X, y,
        k=args.k,
        inner_val_ratio=0.15,
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        warmup_epochs=10,
        save_dir=save_dir,
        save_best=True,
        kern_length=args.kern_length,
        dropout=0.25,
    )
