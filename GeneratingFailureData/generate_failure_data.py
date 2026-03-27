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
N_CRITIC      = 1         # set to 5 for final training run
LAMBDA_GP     = 10
LR            = 1e-4
BETA1, BETA2  = 0.0, 0.9
BATCH_SIZE    = 64
NUM_EPOCHS    = 500

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
# 4 · Gradient penalty
# ─────────────────────────────────────────────────────────────────────────────

def gradient_penalty(
    critic: Critic,
    real:   torch.Tensor,
    fake:   torch.Tensor,
    cond:   torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """WGAN-GP penalty: penalise ||∇critic||₂ ≠ 1 on real/fake interpolations."""
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

    # FIX: flatten() is cleaner than view(B, -1) and handles non-contiguous tensors
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

    FIX vs original: instead of three Python lists that grow with append(),
    we count total windows first, pre-allocate three arrays, then fill them.
    This avoids repeated reallocation and the final np.stack() copy.

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
        n_rows = len(pd.read_csv(os.path.join("dataset/chatter", path)))
        n_win  = max(0, (n_rows - window_size) // step)
        lengths.append(n_win)
        total += n_win

    windows  = np.empty((total, N_CHANNELS, window_size), dtype=np.float32)
    labels   = np.empty(total, dtype=np.int64)
    rpm_idxs = np.empty(total, dtype=np.int64)

    # ── Pass 2: fill pre-allocated arrays ────────────────────────────────
    cursor = 0
    for (path, label, rpm), n_win in zip(filepaths, lengths):
        df  = pd.read_csv(os.path.join("dataset/chatter", path))
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
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    print(f"  Window size : {WINDOW_SIZE}  |  Channels : {N_CHANNELS}")
    print(f"  Noise dim   : {NOISE_DIM}    |  Batch    : {BATCH_SIZE}")

    G = Generator().to(device)
    C = Critic().to(device)

    g_opt = optim.Adam(G.parameters(), lr=LR, betas=(BETA1, BETA2))
    c_opt = optim.Adam(C.parameters(), lr=LR, betas=(BETA1, BETA2))

    for epoch in range(1, NUM_EPOCHS + 1):
        G.train()
        C.train()
        g_epoch, c_epoch, n = 0.0, 0.0, 0

        # FIX: tqdm wraps the loader directly so the bar actually updates.
        # Original created pbar but then iterated `loader` — bar was a no-op.
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}", leave=False)
        for real, labels, rpm_idx in pbar:
            real    = real.to(device,    non_blocking=True)
            labels  = labels.to(device,  non_blocking=True)
            rpm_idx = rpm_idx.to(device, non_blocking=True)

            # FIX: cond is created on device already; no second .to() needed.
            cond = make_condition(labels, rpm_idx)
            B    = real.size(0)

            # ── Critic: N_CRITIC steps per generator step ─────────────────
            for _ in range(N_CRITIC):
                noise  = torch.randn(B, NOISE_DIM, device=device)
                with torch.no_grad():
                    fake = G(noise, cond)     # detach via no_grad — cheaper than .detach()
                gp     = gradient_penalty(C, real, fake.requires_grad_(False), cond, device)
                c_loss = C(fake, cond).mean() - C(real, cond).mean() + LAMBDA_GP * gp
                # NOTE: Wasserstein distance ≈ E[real] - E[fake], so critic
                # loss = E[fake] - E[real] + GP  (we minimise this)

                c_opt.zero_grad()
                c_loss.backward()
                c_opt.step()

            # ── Generator: 1 step ──────────────────────────────────────────
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
                "C": f"{c_loss.item():+.3f}",
                "G": f"{g_loss.item():+.3f}",
            })

        if epoch % 100 == 0:
            avg_c = c_epoch / n
            avg_g = g_epoch / n
            print(f"Epoch [{epoch:>5}/{NUM_EPOCHS}]  "
                  f"W-dist≈{-avg_c:+.4f}   G-loss: {avg_g:+.4f}")
            torch.save(G.state_dict(), f"{checkpoint_dir}/G_{epoch:05d}.pt")
            torch.save(C.state_dict(), f"{checkpoint_dir}/C_{epoch:05d}.pt")

    torch.save(G.state_dict(), f"{checkpoint_dir}/G_final.pt")
    torch.save(C.state_dict(), f"{checkpoint_dir}/C_final.pt")
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
    idx = torch.randperm(len(windows_norm))[:TRAINING_SAMPLES]
    print(f"Training on {TRAINING_SAMPLES} samples")

    loader = build_dataloader(
        windows_norm[idx],
        labels[idx],
        rpm_idxs[idx],
    )

    # ── Train ──────────────────────────────────────────────────────────────
    G, C = train(loader)

    # ── Generate chatter at high RPMs ─────────────────────────────────────
    device    = next(G.parameters()).device
    synth_dfs = []
    for rpm in [7500, 8000, 8500]:
        gen_windows = generate_windows(G, label=1, rpm=rpm, n=300, device=device)
        dfs = windows_to_dataframes(gen_windows)
        for df in dfs:
            df["RPM"]   = rpm
            df["label"] = 1
        synth_dfs.extend(dfs)
        print(f"Generated 300 chatter windows at {rpm} RPM")

    # ── Sanity-check one window ────────────────────────────────────────────
    real_chatter_idx = (labels[idx] == 1).nonzero(as_tuple=True)[0][0].item()
    synth_sample     = generate_windows(G, label=1, rpm=4500, n=1, device=device)[0]
    plot_real_vs_synthetic(windows_norm[idx[real_chatter_idx]].numpy(), synth_sample)

    # ── (Optional) extract features from synthetic windows ─────────────────
    # from your_feature_code import create_windows
    # all_feature_rows = []
    # for df in synth_dfs:
    #     feats = create_windows(df, label=1)
    #     feats["rpm_group"] = df["RPM"].iloc[0]
    #     all_feature_rows.append(feats)
    # synthetic_features = pd.concat(all_feature_rows, ignore_index=True)
    # synthetic_features.to_parquet("synthetic_chatter_features.parquet", index=False)