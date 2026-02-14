# train_cnn.py
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# ----------------------------
# Masked losses / metrics
# ----------------------------
def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # pred/target: (B,C,H,W), mask: (B,1,H,W) or (B,C,H,W)
    if mask.shape[1] == 1:
        mask = mask.expand_as(pred)
    num = (pred - target).abs() * mask
    den = mask.sum().clamp_min(eps)
    return num.sum() / den


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if mask.shape[1] == 1:
        mask = mask.expand_as(pred)
    num = (pred - target).pow(2) * mask
    den = mask.sum().clamp_min(eps)
    return num.sum() / den


@torch.no_grad()
def masked_rmse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(masked_mse(pred, target, mask))


# ----------------------------
# Normalization helpers (masked, NumPy)
# ----------------------------
def _masked_den(mask_mem: np.ndarray) -> float:
    den = float(np.asarray(mask_mem, dtype=np.float64).sum())
    if den <= 0.0:
        raise RuntimeError("Mask has zero support; cannot compute masked statistics.")
    return den


def masked_channel_stats_x(x_mem: np.ndarray, mask_ch: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    x_mem: memmap/ndarray (N,C,H,W) float32
    mask is x_mem[:, mask_ch:mask_ch+1] with values 0/1
    returns (mean[C], std[C]) float32; mask channel forced to mean=0,std=1
    """
    if x_mem.ndim != 4:
        raise ValueError("X must be 4D (N,C,H,W).")
    N, C, _, _ = x_mem.shape
    m = x_mem[:, mask_ch:mask_ch + 1, :, :].astype(np.float64)  # (N,1,H,W)
    den = _masked_den(m)

    mean = np.zeros((C,), dtype=np.float64)
    var = np.zeros((C,), dtype=np.float64)

    for c in range(C):
        xc = x_mem[:, c:c + 1, :, :].astype(np.float64)
        mean[c] = float((xc * m).sum() / den)

    for c in range(C):
        xc = x_mem[:, c:c + 1, :, :].astype(np.float64)
        var[c] = float((((xc - mean[c]) ** 2) * m).sum() / den)

    std = np.sqrt(np.maximum(var, 1e-12))

    mean[mask_ch] = 0.0
    std[mask_ch] = 1.0

    return mean.astype(np.float32), std.astype(np.float32)


def masked_channel_stats_y(y_mem: np.ndarray, mask_mem: np.ndarray, y_channels: Optional[Sequence[int]] = None) -> tuple[np.ndarray, np.ndarray]:
    """
    y_mem: memmap/ndarray (N,Cy,H,W) float32
    mask_mem: memmap/ndarray (N,1,H,W) float32/uint8 0/1
    returns (mean[Cy_sel], std[Cy_sel]) float32
    """
    if y_mem.ndim != 4:
        raise ValueError("Y must be 4D (N,C,H,W).")
    if mask_mem.ndim != 4 or mask_mem.shape[1] != 1:
        raise ValueError("mask_mem must be 4D (N,1,H,W).")

    Y = y_mem if y_channels is None else y_mem[:, list(y_channels), :, :]
    m = mask_mem.astype(np.float64)
    den = _masked_den(m)

    _, C, _, _ = Y.shape
    mean = np.zeros((C,), dtype=np.float64)
    var = np.zeros((C,), dtype=np.float64)

    for c in range(C):
        yc = Y[:, c:c + 1, :, :].astype(np.float64)
        mean[c] = float((yc * m).sum() / den)

    for c in range(C):
        yc = Y[:, c:c + 1, :, :].astype(np.float64)
        var[c] = float((((yc - mean[c]) ** 2) * m).sum() / den)

    std = np.sqrt(np.maximum(var, 1e-12))
    return mean.astype(np.float32), std.astype(np.float32)


# ----------------------------
# Dataset (memmap + normalization)
# ----------------------------
class NpyTensorDataset(Dataset):
    def __init__(
        self,
        x_path: Path,
        y_path: Path,
        y_channels: Optional[Tuple[int, ...]] = None,
        x_mean: Optional[np.ndarray] = None,
        x_std: Optional[np.ndarray] = None,
        y_mean: Optional[np.ndarray] = None,
        y_std: Optional[np.ndarray] = None,
    ):
        self.X = np.load(x_path, mmap_mode="r")  # (N,C,H,W)
        self.Y = np.load(y_path, mmap_mode="r")  # (N,C,H,W)

        if self.X.ndim != 4 or self.Y.ndim != 4:
            raise ValueError("Expected X and Y to be 4D arrays: (N,C,H,W).")
        if self.X.shape[0] != self.Y.shape[0] or self.X.shape[2:] != self.Y.shape[2:]:
            raise ValueError(f"X and Y shape mismatch: X={self.X.shape}, Y={self.Y.shape}")
        if self.X.shape[1] < 1:
            raise ValueError("X must have at least 1 channel (mask in channel 0).")

        if y_channels is not None:
            self.Y = self.Y[:, list(y_channels), :, :]

        self.x_mean = x_mean
        self.x_std = x_std
        self.y_mean = y_mean
        self.y_std = y_std

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        x = torch.from_numpy(np.array(self.X[idx], dtype=np.float32))  # copy slice from memmap
        y = torch.from_numpy(np.array(self.Y[idx], dtype=np.float32))
        m = x[0:1]  # (1,H,W)

        if self.x_mean is not None and self.x_std is not None:
            xm = torch.from_numpy(self.x_mean).view(-1, 1, 1)
            xs = torch.from_numpy(self.x_std).view(-1, 1, 1)
            x = (x - xm) / xs

        y = torch.log10(torch.clamp(y, min=0.0) + 1e-3)

        return x, y, m


# ----------------------------
# Model
# ----------------------------
class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetSmall(nn.Module):
    """
    U-Net for regression:
    input:  (B, c_in, H, W)
    output: (B, c_out, H, W)

    Uses MaxPool downsampling and bilinear upsampling.
    Handles odd sizes by center-cropping skip features.
    """

    def __init__(self, c_in: int, c_out: int, base: int = 32):
        super().__init__()
        self.enc1 = ConvBlock(c_in, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base * 4, base * 8)

        self.up3 = nn.Conv2d(base * 8, base * 4, 1)
        self.dec3 = ConvBlock(base * 8, base * 4)

        self.up2 = nn.Conv2d(base * 4, base * 2, 1)
        self.dec2 = ConvBlock(base * 4, base * 2)

        self.up1 = nn.Conv2d(base * 2, base, 1)
        self.dec1 = ConvBlock(base * 2, base)

        self.out = nn.Conv2d(base, c_out, 1)

    @staticmethod
    def _center_crop(x: torch.Tensor, target_hw) -> torch.Tensor:
        th, tw = target_hw
        h, w = x.shape[-2], x.shape[-1]
        dh = h - th
        dw = w - tw
        if dh == 0 and dw == 0:
            return x
        top = dh // 2
        left = dw // 2
        return x[..., top : top + th, left : left + tw]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Bottleneck
        b = self.bottleneck(self.pool(e3))

        # Decoder stage 3
        u3 = F.interpolate(b, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        u3 = self.up3(u3)
        e3c = self._center_crop(e3, u3.shape[-2:])
        d3 = self.dec3(torch.cat([u3, e3c], dim=1))

        # Decoder stage 2
        u2 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self.up2(u2)
        e2c = self._center_crop(e2, u2.shape[-2:])
        d2 = self.dec2(torch.cat([u2, e2c], dim=1))

        # Decoder stage 1
        u1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self.up1(u1)
        e1c = self._center_crop(e1, u1.shape[-2:])
        d1 = self.dec1(torch.cat([u1, e1c], dim=1))

        return self.out(d1)


# ----------------------------
# Config
# ----------------------------
@dataclass
class TrainConfig:
    train_x: str
    train_y: str
    test_x: str
    test_y: str
    run_dir: str

    y_channels: str  # e.g. "0,1" or "all"
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float

    width: int
    depth: int
    seed: int
    device: str


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_channels(s: str, c_out_total: int) -> Optional[Tuple[int, ...]]:
    s = s.strip().lower()
    if s in ("all", "*"):
        return None
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    ch = tuple(int(p) for p in parts)
    if any((c < 0 or c >= c_out_total) for c in ch):
        raise ValueError(f"y_channels out of bounds. Got {ch}, total C_out={c_out_total}")
    return ch


def save_json(path: Path, obj: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    abs_sum = 0.0
    sq_sum = 0.0
    den_sum = 0.0

    for x, y, m in loader:
        x = x.to(device)
        y = y.to(device)
        m = m.to(device)  # (B,1,H,W)

        pred = model(x)
        pred = pred * m  # enforce gaps = 0

        mm = m.expand_as(pred)
        diff = (pred - y) * mm

        abs_sum += float(diff.abs().sum().item())
        sq_sum += float((diff * diff).sum().item())
        den_sum += float(mm.sum().item())

    if den_sum <= 0.0:
        return {"mae": float("nan"), "rmse": float("nan")}

    mae = abs_sum / den_sum
    rmse = (sq_sum / den_sum) ** 0.5
    return {"mae": mae, "rmse": rmse}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True, help="Folder containing global_X_img_train.npy and global_Y_img_train.npy")
    ap.add_argument("--test_dir", required=True, help="Folder containing global_X_img_test.npy and global_Y_img_test.npy")
    ap.add_argument("--run_dir", required=True, help="Where to write checkpoints and logs")

    ap.add_argument("--y_channels", default="0,1", help='Target channels, e.g. "0,1" for te/ti, or "all"')
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)

    ap.add_argument("--width", type=int, default=64)   # kept for CLI compatibility
    ap.add_argument("--depth", type=int, default=6)    # kept for CLI compatibility
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = ap.parse_args()

    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    train_x = train_dir / "global_X_img_train.npy"
    train_y = train_dir / "global_Y_img_train.npy"
    test_x = test_dir / "global_X_img_test.npy"
    test_y = test_dir / "global_Y_img_test.npy"

    if not train_x.exists() or not train_y.exists() or not test_x.exists() or not test_y.exists():
        raise FileNotFoundError("Expected tensor files named global_X_img_{split}.npy and global_Y_img_{split}.npy in the given dirs.")

    # Quick read shapes
    X0 = np.load(train_x, mmap_mode="r")
    Y0 = np.load(train_y, mmap_mode="r")
    if X0.ndim != 4 or Y0.ndim != 4:
        raise ValueError("Train tensors must be 4D (N,C,H,W).")

    c_in = int(X0.shape[1])
    c_out_total = int(Y0.shape[1])
    y_ch = parse_channels(args.y_channels, c_out_total)
    c_out = c_out_total if y_ch is None else len(y_ch)

    cfg = TrainConfig(
        train_x=str(train_x),
        train_y=str(train_y),
        test_x=str(test_x),
        test_y=str(test_y),
        run_dir=str(run_dir),
        y_channels=args.y_channels,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        width=int(args.width),
        depth=int(args.depth),
        seed=int(args.seed),
        device=str(args.device),
    )
    save_json(run_dir / "config.json", asdict(cfg))

    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    # Compute normalization stats on TRAIN only (masked)
    X_train_mem = np.load(train_x, mmap_mode="r")
    x_mean, x_std = masked_channel_stats_x(X_train_mem, mask_ch=0)
    np.savez_compressed(run_dir / "norm_stats.npz", x_mean=x_mean, x_std=x_std)

    ds_train = NpyTensorDataset(train_x, train_y, y_channels=y_ch, x_mean=x_mean, x_std=x_std)
    ds_test  = NpyTensorDataset(test_x,  test_y,  y_channels=y_ch, x_mean=x_mean, x_std=x_std)


    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    dl_test = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    model = UNetSmall(c_in=c_in, c_out=c_out, base=32).to(device)
    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = {"train_mae": [], "test_mae": [], "train_rmse": [], "test_rmse": []}
    best_test_mae = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        abs_sum = 0.0
        sq_sum = 0.0
        den_sum = 0.0

        for x, y, m in dl_train:
            x = x.to(device)
            y = y.to(device)
            m = m.to(device)

            pred = model(x)
            pred = pred * m  # enforce gaps = 0

            loss = masked_mse(pred, y, m)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            with torch.no_grad():
                mm = m.expand_as(pred)
                diff = (pred - y) * mm
                abs_sum += float(diff.abs().sum().item())
                sq_sum += float((diff * diff).sum().item())
                den_sum += float(mm.sum().item())

        if den_sum <= 0.0:
            train_mae = float("nan")
            train_rmse = float("nan")
        else:
            train_mae = abs_sum / den_sum
            train_rmse = (sq_sum / den_sum) ** 0.5

        test_metrics = evaluate(model, dl_test, device=device)

        history["train_mae"].append(train_mae)
        history["train_rmse"].append(train_rmse)
        history["test_mae"].append(test_metrics["mae"])
        history["test_rmse"].append(test_metrics["rmse"])

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": opt.state_dict(),
            "c_in": c_in,
            "c_out": c_out,
            "y_channels": cfg.y_channels,
            "base": 32,
            "arch": "unet_small",
            "x_mean": x_mean,
            "x_std": x_std,
            "log_eps": 1e-3,
        }
        torch.save(ckpt, run_dir / "checkpoint_last.pt")

        if test_metrics["mae"] < best_test_mae:
            best_test_mae = test_metrics["mae"]
            torch.save(ckpt, run_dir / "checkpoint_best.pt")

        save_json(run_dir / "metrics.json", {"best_test_mae": best_test_mae, **history})

        print(
            f"epoch {epoch:03d} | "
            f"train MAE {train_mae:.6g} RMSE {train_rmse:.6g} | "
            f"test MAE {test_metrics['mae']:.6g} RMSE {test_metrics['rmse']:.6g}"
        )


if __name__ == "__main__":
    main()
