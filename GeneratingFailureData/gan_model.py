"""
Conditional WGAN-GP · Raw Time-Series Chatter Generation 
==================================================================
PIPELINE
  1. Load CSVs  →  slice into fixed-length windows  →  normalise
  2. Train this GAN on all windows
  3. Generate chatter windows at any RPM (including high-RPM gaps)
  4. Optionally run create_windows() on synthetic signals for feature-level augmentation
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# 0 · Constants & config
# ─────────────────────────────────────────────────────────────────────────────

WINDOW_SIZE   = 200
N_CHANNELS    = 3
NOISE_DIM     = 128
N_CRITIC      = 3        
LAMBDA_GP     = 10
LR_G          = 8e-5
LR_C          = 1.5e-4
BETA1, BETA2  = 0.0, 0.9
BATCH_SIZE    = 64
NUM_EPOCHS    = 300
EARLY_STOP_PATIENCE = 40   # epochs without W-dist improvement
MIN_EPOCHS          = 100  # don't stop before the model has had time to warm up

TRAINING_SAMPLES  = 10_000
GEN_BATCH_SIZE    = 256    # max windows per forward pass in generate_windows
NUM_WORKERS       = 8

RPM_CLASSES   = [4500, 5500, 6000, 7500, 8000, 8500]
NUM_RPM       = len(RPM_CLASSES)
NUM_LABELS    = 2
COND_DIM      = NUM_LABELS + NUM_RPM   # 8

rpm_to_idx    = {rpm: i for i, rpm in enumerate(RPM_CLASSES)}

CHANNELS_G    = [256, 128, 64, 32]
CHANNELS_C    = [32, 64, 128, 256]


# ─────────────────────────────────────────────────────────────────────────────
# 1 · Conditioning helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_condition(labels: torch.Tensor, rpm_idx: torch.Tensor) -> torch.Tensor:
    """One-hot encode label + RPM → (B, COND_DIM)."""
    B   = labels.size(0)
    dev = labels.device
    loh = torch.zeros(B, NUM_LABELS, device=dev).scatter_(1, labels.unsqueeze(1), 1.0)
    roh = torch.zeros(B, NUM_RPM,    device=dev).scatter_(1, rpm_idx.unsqueeze(1), 1.0)
    return torch.cat([loh, roh], dim=1)          # (B, 8)


def cond_to_channel(cond: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Broadcast (B, COND_DIM) → (B, COND_DIM, seq_len) for 1-D conv concat."""
    return cond.unsqueeze(-1).expand(-1, -1, seq_len)


# ─────────────────────────────────────────────────────────────────────────────
# 2 · Generator
# ─────────────────────────────────────────────────────────────────────────────

class ResBlock1D(nn.Module):
    """
    Residual block with conditioning injected BEFORE BatchNorm.

    FIX vs original: in the original, condition was concatenated onto the
    feature map *after* the residual block, so BatchNorm could suppress it.
    Now we:
      1. Concatenate condition at the *input* of the block (in channels = C + COND_DIM)
      2. Project back to C channels with the first conv
      3. BatchNorm + second conv operate entirely on the projected features
    This keeps conditioning information in the gradient path through the norm.
    """
    def __init__(self, channels: int):
        super().__init__()
        in_ch = channels + COND_DIM   # condition injected at input
        self.block = nn.Sequential(
            nn.Conv1d(in_ch,     channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
        )
        # 1×1 projection so the skip connection matches the output shape
        self.skip = nn.Conv1d(in_ch, channels, kernel_size=1)
        self.act  = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        c  = cond_to_channel(cond, x.size(2))
        xc = torch.cat([x, c], dim=1)           # (B, C+COND_DIM, L)
        return self.act(self.skip(xc) + self.block(xc))


class Generator(nn.Module):
    """Noise + condition → (B, N_CHANNELS, WINDOW_SIZE) waveform."""

    def __init__(self):
        super().__init__()
        n_up       = len(CHANNELS_G) - 1        # 3 doublings → ×8
        self.start = WINDOW_SIZE // (2 ** n_up) # 200 // 8 = 25
        ch0        = CHANNELS_G[0]

        self.proj = nn.Linear(NOISE_DIM + COND_DIM, ch0 * self.start)

        up_blocks = []
        for i in range(n_up):
            in_ch  = CHANNELS_G[i] + COND_DIM
            out_ch = CHANNELS_G[i + 1]
            up_blocks.append(nn.Sequential(
                nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            ))
        self.up_blocks = nn.ModuleList(up_blocks)

        # ResBlock now receives condition explicitly (see fix above)
        self.res = ResBlock1D(CHANNELS_G[-1])

        # FIX: LazyConv1d is convenient but can hide shape bugs during
        # debugging. Replace with an explicit Conv1d now that we know dims.
        # Input channels = CHANNELS_G[-1] + COND_DIM after res output + cond concat.
        self.out_conv = nn.Conv1d(CHANNELS_G[-1] + COND_DIM, N_CHANNELS, kernel_size=1)

    def forward(self, noise: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B = noise.size(0)

        x = self.proj(torch.cat([noise, cond], dim=1))
        x = x.view(B, CHANNELS_G[0], self.start)

        for up in self.up_blocks:
            c = cond_to_channel(cond, x.size(2))
            x = torch.cat([x, c], dim=1)
            x = up(x)

        # ResBlock handles its own conditioning internally now
        x = self.res(x, cond)

        # Final projection
        c   = cond_to_channel(cond, x.size(2))
        out = torch.tanh(self.out_conv(torch.cat([x, c], dim=1)))
        return out   # (B, N_CHANNELS, WINDOW_SIZE)


# ─────────────────────────────────────────────────────────────────────────────
# 3 · Critic
# ─────────────────────────────────────────────────────────────────────────────

class Critic(nn.Module):
    """(B, N_CHANNELS, WINDOW_SIZE) + condition → scalar Wasserstein score."""

    def __init__(self):
        super().__init__()
        down_blocks = []
        in_ch = N_CHANNELS + COND_DIM
        for out_ch in CHANNELS_C:
            down_blocks.append(nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                # No BatchNorm in critic — it destabilises gradient penalty.
            ))
            in_ch = out_ch + COND_DIM

        self.down_blocks = nn.ModuleList(down_blocks)
        self.pool        = nn.AdaptiveAvgPool1d(1)
        self.fc          = nn.Linear(CHANNELS_C[-1], 1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        c = cond_to_channel(cond, x.size(2))
        x = torch.cat([x, c], dim=1)

        for i, block in enumerate(self.down_blocks):
            x = block(x)
            if i < len(self.down_blocks) - 1:
                c = cond_to_channel(cond, x.size(2))
                x = torch.cat([x, c], dim=1)

        return self.fc(self.pool(x).squeeze(-1))   # (B, 1)

# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint and CSV Functions
# ─────────────────────────────────────────────────────────────────────────────
def save_windows_to_csv(
    windows:    np.ndarray,
    labels:     list[int] | np.ndarray,
    rpms:       list[int] | np.ndarray,
    output_dir: str = "dataset/synthetic",
) -> None:
    """
    Save generated (n, 3, WINDOW_SIZE) windows to CSV files that mirror
    the format of your real input files.

    Each file is named:  {rpm}_{label}_synth_{i}.csv
    Each file has columns: T, X, Y, Z  — same as your real CSVs.

    Parameters
    ----------
    windows    : (n, 3, WINDOW_SIZE) float32 array from generate_windows()
    labels     : length-n sequence of integer labels (0 or 1)
    rpms       : length-n sequence of RPM values
    output_dir : folder to write into (created if it doesn't exist)
    """
    os.makedirs(output_dir, exist_ok=True)
    t_col = np.arange(windows.shape[2])

    # Count existing files per (rpm, label) pair so indices don't collide
    # across multiple calls
    counters: dict[tuple[int,int], int] = {}

    for i, (window, label, rpm) in enumerate(zip(windows, labels, rpms)):
        key = (int(rpm), int(label))
        idx = counters.get(key, 0)
        counters[key] = idx + 1

        fname = f"{rpm}_{label}_synth_{idx:04d}.csv"
        fpath = os.path.join(output_dir, fname)

        df = pd.DataFrame(
            window.T,                          # (WINDOW_SIZE, 3)
            columns=["X", "Y", "Z"],
        )
        df.insert(0, "T", t_col)
        df.to_csv(fpath, index=False)

    print(f"Saved {len(windows)} windows to {output_dir}/")


def save_checkpoint(
    G:              "Generator",
    C:              "Critic",
    g_opt:          torch.optim.Optimizer,
    c_opt:          torch.optim.Optimizer,
    epoch:          int,
    checkpoint_dir: str = "GeneratingFailureData/checkpoints",
) -> None:
    """
    Save a full resumable checkpoint — model weights + optimiser states + epoch.
    This lets you resume training exactly where you left off, including the
    Adam moment buffers (loss of those = first few epochs after resume are noisy).
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch:05d}.pt")
    torch.save({
        "epoch":       epoch,
        "G_state":     G.state_dict(),
        "C_state":     C.state_dict(),
        "g_opt_state": g_opt.state_dict(),
        "c_opt_state": c_opt.state_dict(),
    }, path)
    print(f"✓ Checkpoint saved → {path}")


def load_checkpoint(
    G:              "Generator",
    C:              "Critic",
    g_opt:          torch.optim.Optimizer,
    c_opt:          torch.optim.Optimizer,
    path:           str,
    device:         torch.device,
) -> int:
    """
    Load a checkpoint saved by save_checkpoint().
    Restores weights + optimiser states in-place, returns the saved epoch
    so your training loop can resume from the right number.

    Usage
    -----
    start_epoch = load_checkpoint(G, C, g_opt, c_opt, "checkpoints/checkpoint_00500.pt", device)
    for epoch in range(start_epoch + 1, NUM_EPOCHS + 1):
        ...
    """
    ckpt = torch.load(path, map_location=device)
    G.load_state_dict(ckpt["G_state"])
    C.load_state_dict(ckpt["C_state"])
    g_opt.load_state_dict(ckpt["g_opt_state"])
    c_opt.load_state_dict(ckpt["c_opt_state"])
    epoch = ckpt["epoch"]
    print(f"Resumed from epoch {epoch}  ← {path}")
    return epoch

# ─────────────────────────────────────────────────────────────────────────────
# 4 · Gradient penalty
# ─────────────────────────────────────────────────────────────────────────────

def gradient_penalty(
    critic: Critic,
    real:   torch.Tensor,
    fake:   torch.Tensor,
    cond:   torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """WGAN-GP penalty on CLEAN interpolations — noise must NOT be applied here."""
    B     = real.size(0)
    alpha = torch.rand(B, 1, 1, device=device).expand_as(real)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)

    score = critic(interp, cond)
    grad  = torch.autograd.grad(
        outputs=score,
        inputs=interp,
        grad_outputs=torch.ones_like(score),
        create_graph=True,
        retain_graph=True,
    )[0]

    grad_norm = grad.flatten(start_dim=1).norm(2, dim=1)
    return ((grad_norm - 1) ** 2).mean()

# ─────────────────────────────────────────────────────────────────────────────
# 5 · Data preparation
# ─────────────────────────────────────────────────────────────────────────────

def load_csv_to_windows(
    filepaths:   list[tuple[str, int, int]],
    window_size: int   = WINDOW_SIZE,
    overlap:     float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load raw CSVs and slice into overlapping windows.

    Returns
    -------
    windows  : float32  (N, N_CHANNELS, window_size)
    labels   : int64    (N,)
    rpm_idxs : int64    (N,)
    """
    step = int(window_size * (1 - overlap))

    # ── Pass 1: count windows so we can pre-allocate ──────────────────────
    total = 0
    lengths: list[int] = []
    for path, _, _ in filepaths:
        n_rows = len(pd.read_csv(os.path.join("dataset", "chatter", path)))
        n_win  = max(0, (n_rows - window_size) // step)
        lengths.append(n_win)
        total += n_win

    windows  = np.empty((total, N_CHANNELS, window_size), dtype=np.float32)
    labels   = np.empty(total, dtype=np.int64)
    rpm_idxs = np.empty(total, dtype=np.int64)

    # ── Pass 2: fill pre-allocated arrays ────────────────────────────────
    cursor = 0
    for (path, label, rpm), n_win in zip(filepaths, lengths):
        df  = pd.read_csv(os.path.join("dataset", "chatter", path))
        sig = df[["X", "Y", "Z"]].to_numpy(dtype=np.float32)   # (T, 3)

        # Build all start indices at once, then slice with a vectorised index
        starts = np.arange(n_win) * step                        # (n_win,)
        idx    = starts[:, None] + np.arange(window_size)       # (n_win, window_size)
        wins   = sig[idx]                                        # (n_win, window_size, 3)

        windows[cursor : cursor + n_win]  = wins.transpose(0, 2, 1)  # → (n_win, 3, W)
        labels[cursor : cursor + n_win]   = label
        rpm_idxs[cursor : cursor + n_win] = rpm_to_idx[rpm]
        cursor += n_win

    return windows, labels, rpm_idxs


def per_window_normalise(
    windows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalise each window independently to [-1, 1].

    FIX vs original: now operates on a torch.Tensor throughout so the caller
    doesn't need to convert; uses .amin/.amax which are dim-tuple aware.

    Returns
    -------
    normed  : (N, 3, W) float32 tensor in [-1, 1]
    w_min   : (N, 3, 1)
    w_range : (N, 3, 1)
    """
    w_min   = windows.amin(dim=2, keepdim=True)    # (N, 3, 1)
    w_max   = windows.amax(dim=2, keepdim=True)
    w_range = (w_max - w_min).clamp(min=1e-8)      # avoid div-by-zero
    normed  = 2.0 * (windows - w_min) / w_range - 1.0
    return normed, w_min, w_range


def denormalise(
    normed:  torch.Tensor,
    w_min:   torch.Tensor,
    w_range: torch.Tensor,
) -> torch.Tensor:
    """Invert per_window_normalise. Accepts torch.Tensor or np.ndarray."""
    return (normed + 1.0) / 2.0 * w_range + w_min


def build_dataloader(
    windows:  torch.Tensor,
    labels:   torch.Tensor,
    rpm_idxs: torch.Tensor,
) -> DataLoader:
    dataset = TensorDataset(windows, labels, rpm_idxs)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6 · Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(loader: DataLoader, checkpoint_dir: str = "GeneratingFailureData/checkpoints"):
    """Train the Conditional WGAN-GP. Returns trained Generator and Critic."""

    checkpoint_dir = os.path.abspath(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {checkpoint_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    print(f"  Window size : {WINDOW_SIZE}  |  Channels : {N_CHANNELS}")
    print(f"  Noise dim   : {NOISE_DIM}    |  Batch    : {BATCH_SIZE}")

    G = Generator().to(device)
    C = Critic().to(device)

    g_opt = optim.Adam(G.parameters(), lr=LR_G, betas=(BETA1, BETA2))
    c_opt = optim.Adam(C.parameters(), lr=LR_C, betas=(BETA1, BETA2))

    g_sched = optim.lr_scheduler.CosineAnnealingLR(g_opt, T_max=NUM_EPOCHS, eta_min=1e-5)
    c_sched = optim.lr_scheduler.CosineAnnealingLR(c_opt, T_max=NUM_EPOCHS, eta_min=1e-5)

    best_wdist        = -float("inf")  # ← must be negative inf, not positive
    best_epoch        = -1
    epochs_no_improve = 0
    # Instance noise decays to zero by halfway through training.
    # This gives the critic a noisy curriculum early on (harder to memorise)
    # then clean signals later (accurate GP gradients).
    noise_decay_end = NUM_EPOCHS * 0.5

    for epoch in range(1, NUM_EPOCHS + 1):
        G.train()
        C.train()
        g_epoch, c_epoch, n = 0.0, 0.0, 0

        # ── Decaying instance noise std ───────────────────────────────────
        # Linearly from 0.05 → 0.0 over the first half of training
        noise_std = 0.05 * max(0.0, 1.0 - epoch / noise_decay_end)

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}", leave=False)
        for real, labels, rpm_idx in pbar:
            real    = real.to(device, non_blocking=True)
            labels  = labels.to(device, non_blocking=True)
            rpm_idx = rpm_idx.to(device, non_blocking=True)
            cond    = make_condition(labels, rpm_idx)
            B       = real.size(0)

            # ── Critic ────────────────────────────────────────────────────
            for _ in range(N_CRITIC):
                noise = torch.randn(B, NOISE_DIM, device=device)

                G.eval()
                with torch.no_grad():
                    fake_clean = G(noise, cond)
                G.train()

                # Instance noise on BOTH real and fake for critic input.
                # GP always uses clean tensors so gradients stay accurate.
                if noise_std > 0:
                    real_noisy = real + noise_std * torch.randn_like(real)
                    fake_noisy = fake_clean + noise_std * torch.randn_like(fake_clean)
                else:
                    real_noisy = real
                    fake_noisy = fake_clean

                gp     = gradient_penalty(C, real, fake_clean, cond, device)
                c_loss = (
                    C(fake_noisy, cond).mean()
                    - C(real_noisy, cond).mean()
                    + LAMBDA_GP * gp
                )

                c_opt.zero_grad()
                c_loss.backward()
                c_opt.step()

            # ── Generator ─────────────────────────────────────────────────
            noise  = torch.randn(B, NOISE_DIM, device=device)
            fake   = G(noise, cond)
            g_loss = -C(fake, cond).mean()

            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()

            c_epoch += c_loss.item()
            g_epoch += g_loss.item()
            n       += 1

            pbar.set_postfix({
                "C":    f"{c_loss.item():+.3f}",
                "G":    f"{g_loss.item():+.3f}",
                "σ":    f"{noise_std:.3f}",
            })

        g_sched.step()
        c_sched.step()

        # ── Per-epoch W-dist check ─────────
        avg_c  = c_epoch / n
        avg_g  = g_epoch / n
        wdist  = -avg_c

        # ── Logging every 50 epochs ───────────────────────────────────────
        if epoch % 50 == 0:
            print(f"Epoch [{epoch:>5}/{NUM_EPOCHS}]  "
                  f"W-dist≈{wdist:+.4f}   G-loss: {avg_g:+.4f}   σ={noise_std:.4f}")
        
        # ── Early stopping based on W-dist improvement ────────────────────
        if wdist > best_wdist:
            best_wdist        = wdist
            best_epoch        = epoch
            epochs_no_improve = 0
            save_checkpoint(G, C, g_opt, c_opt, epoch, checkpoint_dir)
        else:
            epochs_no_improve += 1

        if epoch >= MIN_EPOCHS and epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stop at epoch {epoch} — no improvement for {EARLY_STOP_PATIENCE} epochs.")
            break    

    # Only save a final checkpoint if training ended naturally and was valid
    print(f"\nTraining complete. Best model: epoch {best_epoch}, W-dist={best_wdist:.4f}")
    print(f"Load it with: load_checkpoint(G, C, g_opt, c_opt, 'checkpoints/checkpoint_epoch_{best_epoch:05d}.pt', device)")
    
    print("Done.")
    return G, C


# ─────────────────────────────────────────────────────────────────────────────
# 7 · Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_windows(
    G:      Generator,
    label:  int,
    rpm:    int,
    n:      int = 200,
    device: torch.device | None = None,
) -> np.ndarray:
    """
    Generate n synthetic (X, Y, Z) windows for a given label + RPM.

    FIX vs original:
      - G.eval() is paired with a G.train() restore via try/finally so
        generate_windows() is safe to call during a training run.
      - Large n is batched across GEN_BATCH_SIZE to avoid OOM.

    Returns float32 array (n, N_CHANNELS, WINDOW_SIZE) in [-1, 1] space.
    """
    if device is None:
        device = next(G.parameters()).device

    was_training = G.training
    G.eval()
    try:
        parts = []
        remaining = n
        with torch.no_grad():
            while remaining > 0:
                bs       = min(remaining, GEN_BATCH_SIZE)
                labels_t = torch.full((bs,), label,           dtype=torch.long, device=device)
                rpm_t    = torch.full((bs,), rpm_to_idx[rpm], dtype=torch.long, device=device)
                cond     = make_condition(labels_t, rpm_t)
                noise    = torch.randn(bs, NOISE_DIM, device=device)
                parts.append(G(noise, cond).cpu())
                remaining -= bs
        windows = torch.cat(parts, dim=0).numpy()   # (n, 3, WINDOW_SIZE)
    finally:
        if was_training:
            G.train()

    return windows


def windows_to_dataframes(windows: np.ndarray) -> list[pd.DataFrame]:
    """
    Convert (n, 3, WINDOW_SIZE) array → list of DataFrames [T, X, Y, Z].

    FIX vs original: vectorised transpose + column assignment; the only
    remaining Python loop is the unavoidable per-DataFrame construction,
    but we avoid re-creating the T column inside the loop.
    """
    n, _, W = windows.shape
    t_col   = np.arange(W)                      # shared across all frames
    xyz     = windows.transpose(0, 2, 1)        # (n, W, 3)  — vectorised

    return [
        pd.DataFrame(
            np.concatenate([t_col[:, None], xyz[i]], axis=1),
            columns=["T", "X", "Y", "Z"],
        )
        for i in range(n)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 8 · Quick visual sanity check
# ─────────────────────────────────────────────────────────────────────────────

def plot_real_vs_synthetic(real_window: np.ndarray, synth_window: np.ndarray):
    """
    Side-by-side plot of one real and one synthetic (X, Y, Z) window.
    real_window / synth_window: (3, WINDOW_SIZE) arrays.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot")
        return

    fig, axes = plt.subplots(3, 2, figsize=(12, 6), sharex=True)
    ch_labels = ["X", "Y", "Z"]
    for i, ax_row in enumerate(axes):
        ax_row[0].plot(real_window[i],  color="steelblue",  lw=0.8)
        ax_row[1].plot(synth_window[i], color="darkorange", lw=0.8)
        ax_row[0].set_ylabel(ch_labels[i])

    axes[0][0].set_title("Real")
    axes[0][1].set_title("Synthetic")
    plt.tight_layout()
    plt.savefig("real_vs_synthetic.png", dpi=150)
    plt.show()
    print("Saved → real_vs_synthetic.png")


# ─────────────────────────────────────────────────────────────────────────────
# 9 · MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    file_manifest = [
        ("4500_1.csv",  1, 4500),
        ("5500_1.csv",  1, 5500),
        ("5500_3.csv",  1, 5500),
        ("6000_1.csv",  1, 6000),
        ("7500_1.csv",  0, 7500),
        ("7500_2.csv",  0, 7500),
        ("7500_3.csv",  0, 7500),
        ("8000_1.csv",  0, 8000),
        ("8000_4.csv",  0, 8000),
        ("8500_1.csv",  0, 8500),
    ]

    # ── Load + window ──────────────────────────────────────────────────────
    print("Loading Data and Slicing into Windows...")
    windows_np, labels_np, rpm_idxs_np = load_csv_to_windows(file_manifest)

    # Convert to tensors once; all downstream ops stay in torch
    windows  = torch.from_numpy(windows_np)
    labels   = torch.from_numpy(labels_np)
    rpm_idxs = torch.from_numpy(rpm_idxs_np)

    # ── Normalise ──────────────────────────────────────────────────────────
    windows_norm, w_min, w_range = per_window_normalise(windows)

    print(f"Total dataset: {len(windows_norm)} windows  "
        f"| chatter: {(labels == 1).sum().item()} "
        f"| no-chatter: {(labels == 0).sum().item()}")

    # ── Subsample for iterative tuning runs ───────────────────────────────
    chatter_idx    = (labels == 1).nonzero(as_tuple=True)[0]
    nochatter_idx  = (labels == 0).nonzero(as_tuple=True)[0]

    n_each = TRAINING_SAMPLES // 2

    sampled = torch.cat([
        chatter_idx[torch.randperm(len(chatter_idx))[:n_each]],
        nochatter_idx[torch.randperm(len(nochatter_idx))[:n_each]],
    ])

    # (optional but recommended) shuffle after concat
    sampled = sampled[torch.randperm(len(sampled))]

    print(f"Training on {len(sampled)} samples")

    # ── Build dataloader using sampled indices ────────────────────────────
    loader = build_dataloader(
        windows_norm[sampled],
        labels[sampled],
        rpm_idxs[sampled],  
    )

    # ── Train ──────────────────────────────────────────────────────────────
    print("Starting training...")
    G, C = train(loader)

    # ── Generate chatter at high RPMs ─────────────────────────────────────
    print("Generating synthetic chatter windows at high RPMs...")
    device    = next(G.parameters()).device
    synth_dfs = []

    all_windows_list = []
    all_labels_list = []
    all_rpms_list = []

    for rpm in [7500, 8000, 8500]:
        gen_windows = generate_windows(G, label=1, rpm=rpm, n=300, device=device)
        dfs = windows_to_dataframes(gen_windows)
        for df in dfs:
            df["RPM"] = rpm
            df["label"] = 1
        synth_dfs.extend(dfs)

        all_windows_list.append(gen_windows)
        all_labels_list.append(np.ones(300, dtype=int))
        all_rpms_list.append(np.full(300, rpm, dtype=int))
        print(f"Generated 300 chatter windows at {rpm} RPM")

    all_windows = np.concatenate(all_windows_list)
    all_labels  = np.concatenate(all_labels_list)
    all_rpms    = np.concatenate(all_rpms_list)

    save_windows_to_csv(all_windows, all_labels, all_rpms, output_dir="dataset/synthetic")

    # ── Sanity-check one window ────────────────────────────────────────────
    sampled_labels = labels[sampled]
    sampled_windows = windows_norm[sampled]

    real_chatter_idx = (sampled_labels == 1).nonzero(as_tuple=True)[0][0].item()

    synth_sample = generate_windows(G, label=1, rpm=4500, n=1, device=device)[0]

    plot_real_vs_synthetic(
        sampled_windows[real_chatter_idx].numpy(),  
        synth_sample
    )

    # ── (Optional) extract features from synthetic windows ─────────────────
    # from your_feature_code import create_windows
    # all_feature_rows = []
    # for df in synth_dfs:
    #     feats = create_windows(df, label=1)
    #     feats["rpm_group"] = df["RPM"].iloc[0]
    #     all_feature_rows.append(feats)
    # synthetic_features = pd.concat(all_feature_rows, ignore_index=True)
    # synthetic_features.to_parquet("synthetic_chatter_features.parquet", index=False)