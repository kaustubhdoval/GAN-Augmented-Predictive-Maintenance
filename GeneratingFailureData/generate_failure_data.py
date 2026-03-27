"""
Conditional WGAN-GP · Raw Time-Series Chatter Generation
==================================================================
Generates synthetic (T, X, Y, Z) windows conditioned on:
  - label    : 0 = no-chatter, 1 = chatter
  - rpm_group: one of [4500, 5500, 6000, 7500, 8000, 8500]

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


# ─────────────────────────────────────────────────────────────────────────────
# 0 · Constants & config
# ─────────────────────────────────────────────────────────────────────────────

WINDOW_SIZE   = 200        # timesteps per window  (match with create_windows)
N_CHANNELS    = 3          # X, Y, Z  (Time acts as an index)
NOISE_DIM     = 128        # latent vector size
N_CRITIC      = 5          # critic updates per generator update
LAMBDA_GP     = 10         # gradient-penalty weight
LR            = 1e-4
BETA1, BETA2  = 0.0, 0.9   # Adam betas — 0.0 for β1 is standard in WGAN
BATCH_SIZE    = 32         # keep small; chatter data is scarce
NUM_EPOCHS    = 1000

RPM_CLASSES   = [4500, 5500, 6000, 7500, 8000, 8500]
NUM_RPM       = len(RPM_CLASSES)
NUM_LABELS    = 2
COND_DIM      = NUM_LABELS + NUM_RPM   # = 8

rpm_to_idx    = {rpm: i for i, rpm in enumerate(RPM_CLASSES)}

CHANNELS_G    = [256, 128, 64, 32]   # decoder channel progression (wide → narrow)
CHANNELS_C    = [32, 64, 128, 256]   # encoder channel progression (narrow → wide)


# ─────────────────────────────────────────────────────────────────────────────
# 1 · Conditioning helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_condition(labels: torch.Tensor, rpm_idx: torch.Tensor) -> torch.Tensor:
    """
    One-hot encode label + RPM, return shape (B, COND_DIM).
    Separate one-hots so the network learns orthogonal axes for
    'is chatter' vs 'which RPM'.
    """
    B = labels.size(0)
    dev = labels.device
    loh = torch.zeros(B, NUM_LABELS, device=dev)
    loh.scatter_(1, labels.unsqueeze(1), 1.0)
    roh = torch.zeros(B, NUM_RPM, device=dev)
    roh.scatter_(1, rpm_idx.unsqueeze(1), 1.0)
    return torch.cat([loh, roh], dim=1)          # (B, 8)


def cond_to_channel(cond: torch.Tensor, seq_len: int) -> torch.Tensor:
    """
    Broadcast a (B, COND_DIM) condition vector into a (B, COND_DIM, seq_len)
    channel so it can be concatenated with convolutional feature maps at any
    resolution.  This is the standard 'concat-along-channel' conditioning
    trick for 1-D CNNs.
    """
    return cond.unsqueeze(-1).expand(-1, -1, seq_len)


# ─────────────────────────────────────────────────────────────────────────────
# 2 · Generator  (noise → waveform)
# ─────────────────────────────────────────────────────────────────────────────

class ResBlock1D(nn.Module):
    """
    Residual block with two 1-D convolutions and a skip connection.
    Lets the generator refine details without gradient vanishing.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(x + self.block(x))


class Generator(nn.Module):
    """
    Noise + condition  →  (B, N_CHANNELS, WINDOW_SIZE) waveform

    Architecture:
      1. Project noise+cond → dense feature map  (CHANNELS_G[0] × start_len)
      2. Upsample via transposed convolutions until WINDOW_SIZE is reached
      3. Residual block at full resolution to sharpen temporal patterns
      4. 1×1 conv to collapse to N_CHANNELS, Tanh to bound output in [-1, 1]

    start_len is chosen so that log2(WINDOW_SIZE / start_len) upsample steps
    each doubling the sequence, landing exactly on WINDOW_SIZE.
    """
    def __init__(self):
        super().__init__()
        # How many upsample stages do we need?
        n_up       = len(CHANNELS_G) - 1          # 3 doublings  → ×8
        self.start = WINDOW_SIZE // (2 ** n_up)   # 200 // 8 = 25
        ch0        = CHANNELS_G[0]

        # Project noise + condition to a dense sequence
        self.proj = nn.Linear(NOISE_DIM + COND_DIM, ch0 * self.start)

        # Upsampling blocks: each doubles the sequence length
        up_blocks = []
        for i in range(n_up):
            in_ch  = CHANNELS_G[i] + COND_DIM   # concat condition at every scale
            out_ch = CHANNELS_G[i + 1]
            up_blocks.append(nn.Sequential(
                nn.ConvTranspose1d(in_ch, out_ch,
                                   kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            ))
        self.up_blocks = nn.ModuleList(up_blocks)

        # Residual refinement at full resolution
        self.res = ResBlock1D(CHANNELS_G[-1] + COND_DIM)

        # Final projection to signal channels
        # TODO: Do not hardcode this value
        final_channels = 48
        self.out_conv = nn.Conv1d(final_channels, 3, kernel_size=1)

    def forward(self, noise: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B = noise.size(0)

        # Dense projection → reshape to (B, ch0, start_len)
        x = self.proj(torch.cat([noise, cond], dim=1))
        x = x.view(B, CHANNELS_G[0], self.start)

        # Upsample, injecting condition at every stage
        for up in self.up_blocks:
            c = cond_to_channel(cond, x.size(2))
            x = torch.cat([x, c], dim=1)
            x = up(x)

        # Residual block + condition injection
        c = cond_to_channel(cond, x.size(2))
        x = torch.cat([x, c], dim=1)
        x = self.res(x)

        # Collapse to N_CHANNELS, bound to [-1, 1]
        c = cond_to_channel(cond, x.size(2))
        x = torch.cat([x, c], dim=1)
        out = torch.tanh(self.out_conv(x))   # (B, 3, WINDOW_SIZE)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 3 · Critic  (waveform → scalar score)
# ─────────────────────────────────────────────────────────────────────────────

class Critic(nn.Module):
    """
    (B, N_CHANNELS, WINDOW_SIZE) + condition  →  scalar Wasserstein score

    Architecture:
      1. Inject condition as extra channels at input
      2. Strided convolutions to downsample (mirrors generator upsampling)
      3. Global average pool to get a fixed-size vector
      4. Linear → scalar score  (NO sigmoid — this is not a classifier)

    No BatchNorm in critic — it destabilises gradient penalty.
    Use LayerNorm if you need normalisation.
    """
    def __init__(self):
        super().__init__()
        down_blocks = []
        in_ch = N_CHANNELS + COND_DIM
        for out_ch in CHANNELS_C:
            down_blocks.append(nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            ))
            in_ch = out_ch + COND_DIM    # re-inject condition after each block

        self.down_blocks = nn.ModuleList(down_blocks)
        self.pool        = nn.AdaptiveAvgPool1d(1)
        self.fc          = nn.Linear(CHANNELS_C[-1], 1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # Inject condition at input
        c = cond_to_channel(cond, x.size(2))
        x = torch.cat([x, c], dim=1)

        for i, block in enumerate(self.down_blocks):
            x = block(x)
            if i < len(self.down_blocks) - 1:
                # Re-inject at each intermediate scale
                c = cond_to_channel(cond, x.size(2))
                x = torch.cat([x, c], dim=1)

        x = self.pool(x).squeeze(-1)     # (B, CHANNELS_C[-1])
        return self.fc(x)                # (B, 1)


# ─────────────────────────────────────────────────────────────────────────────
# 4 · Gradient penalty
# ─────────────────────────────────────────────────────────────────────────────

def gradient_penalty(critic, real, fake, cond, device):
    """
    WGAN-GP penalty: interpolate real/fake, penalise ||∇critic||₂ ≠ 1.
    Enforces the 1-Lipschitz constraint that makes Wasserstein distance valid.
    """
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

    grad_norm = grad.view(B, -1).norm(2, dim=1)
    return ((grad_norm - 1) ** 2).mean()


# ─────────────────────────────────────────────────────────────────────────────
# 5 · Data preparation
# ─────────────────────────────────────────────────────────────────────────────

def load_csv_to_windows(
    filepaths: list[tuple[str, int, int]],
    window_size: int = WINDOW_SIZE,
    overlap: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load raw CSVs and slice into overlapping windows.

    Parameters
    ----------
    filepaths : list of (csv_path, label, rpm)
                e.g. [("4500_1.csv", 1, 4500), ("7500_1.csv", 0, 7500), ...]
    window_size : timesteps per window
    overlap     : fraction of window to overlap between consecutive windows

    Returns
    -------
    windows  : float32 array  (N, N_CHANNELS, window_size)   — X, Y, Z
    labels   : int64 array    (N,)
    rpm_idxs : int64 array    (N,)
    """
    step = int(window_size * (1 - overlap))
    all_windows, all_labels, all_rpms = [], [], []

    for path, label, rpm in filepaths:
        full_path = os.path.join("dataset/chatter", path)
        df  = pd.read_csv(full_path)
        sig = df[["X", "Y", "Z"]].values.astype(np.float32)   # (T, 3)

        for start in range(0, len(sig) - window_size, step):
            w = sig[start : start + window_size]     # (window_size, 3)
            all_windows.append(w.T)                  # (3, window_size) — channels first
            all_labels.append(label)
            all_rpms.append(rpm_to_idx[rpm])

    windows  = np.stack(all_windows).astype(np.float32)   # (N, 3, window_size)
    labels   = np.array(all_labels,  dtype=np.int64)
    rpm_idxs = np.array(all_rpms,    dtype=np.int64)
    return windows, labels, rpm_idxs


def per_window_normalise(windows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalise each window independently to [-1, 1] using its own min/max.

    Why per-window rather than global?
      - Amplitude varies a lot between RPMs; global norm would let the model
        cheat by reading amplitude to identify RPM instead of waveform shape
      - We save (min, max) per window so we can de-normalise generated windows

    Returns
    -------
    normed  : (N, 3, window_size) scaled to [-1, 1]
    w_min   : (N, 3, 1)  per-window per-channel minimum
    w_range : (N, 3, 1)  per-window per-channel range
    """
    w_min   = windows.min(axis=2, keepdims=True)              # (N, 3, 1)
    w_max   = windows.max(axis=2, keepdims=True)
    w_range = w_max - w_min + 1e-8
    normed  = 2.0 * (windows - w_min) / w_range - 1.0        # → [-1, 1]
    return normed, w_min, w_range


def denormalise(normed: np.ndarray, w_min: np.ndarray, w_range: np.ndarray) -> np.ndarray:
    """Invert per_window_normalise for a batch of windows."""
    return (normed + 1.0) / 2.0 * w_range + w_min


def build_dataloader(
    windows: np.ndarray,
    labels:  np.ndarray,
    rpm_idxs: np.ndarray,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(windows),
        torch.from_numpy(labels),
        torch.from_numpy(rpm_idxs),
    )
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


# ─────────────────────────────────────────────────────────────────────────────
# 6 · Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(loader: DataLoader, checkpoint_dir: str = "GeneratingFailureData/checkpoints"):
    """
    Train the Conditional WGAN-GP.

    Returns trained Generator and Critic.
    """
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
        g_epoch, c_epoch, n = 0.0, 0.0, 0

        for real, labels, rpm_idx in loader:
            real    = real.to(device)
            labels  = labels.to(device)
            rpm_idx = rpm_idx.to(device)
            cond    = make_condition(labels, rpm_idx)
            B       = real.size(0)

            # ── Critic: N_CRITIC steps per generator step ─────────────────
            for _ in range(N_CRITIC):
                noise     = torch.randn(B, NOISE_DIM, device=device)
                fake      = G(noise, cond).detach()
                gp        = gradient_penalty(C, real, fake, cond, device)
                c_loss    = -(C(real, cond).mean() - C(fake, cond).mean()) \
                            + LAMBDA_GP * gp

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
    G:         Generator,
    label:     int,
    rpm:       int,
    n:         int = 200,
    device:    torch.device | None = None,
) -> np.ndarray:
    """
    Generate n synthetic (X, Y, Z) windows for a given label + RPM.

    Returns
    -------
    windows : float32 array (n, 3, WINDOW_SIZE) in normalised [-1, 1] space.
              Call denormalise() with reference stats if you need physical units,
              or feed directly into create_windows() after transposing to (T, 3).
    """
    if device is None:
        device = next(G.parameters()).device
    G.eval()
    with torch.no_grad():
        labels_t  = torch.full((n,), label,          dtype=torch.long, device=device)
        rpm_t     = torch.full((n,), rpm_to_idx[rpm], dtype=torch.long, device=device)
        cond      = make_condition(labels_t, rpm_t)
        noise     = torch.randn(n, NOISE_DIM, device=device)
        windows   = G(noise, cond).cpu().numpy()     # (n, 3, WINDOW_SIZE)
    return windows


def windows_to_dataframes(windows: np.ndarray) -> list[pd.DataFrame]:
    """
    Convert generated (n, 3, WINDOW_SIZE) array into a list of DataFrames
    with columns [T, X, Y, Z] — the same format as your raw CSVs.
    You can then call create_windows() on each one.
    """
    dfs = []
    for w in windows:
        df = pd.DataFrame(w.T, columns=["X", "Y", "Z"])
        df.insert(0, "T", np.arange(len(df)))
        dfs.append(df)
    return dfs


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
    labels = ["X", "Y", "Z"]
    for i, ax_row in enumerate(axes):
        ax_row[0].plot(real_window[i],  color="steelblue",  lw=0.8)
        ax_row[1].plot(synth_window[i], color="darkorange", lw=0.8)
        ax_row[0].set_ylabel(labels[i])

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

    # ── Step 1: specify your files ─────────────────────────────────────────
    file_manifest = [
        # (path,            label,  rpm  )
        ("4500_1.csv",      1,      4500),
        ("5500_1.csv",      1,      5500),
        ("5500_3.csv",      1,      5500),
        ("6000_1.csv",      1,      6000),
        ("7500_1.csv",      0,      7500),
        ("7500_2.csv",      0,      7500),
        ("7500_3.csv",      0,      7500),
        ("8000_1.csv",      0,      8000),
        ("8000_4.csv",      0,      8000),
        ("8500_1.csv",      0,      8500),
    ]

    # ── Step 2: load + window + normalise ──────────────────────────────────
    windows, labels, rpm_idxs = load_csv_to_windows(file_manifest)
    windows_norm, w_min, w_range = per_window_normalise(windows)

    print(f"Dataset: {len(windows)} windows "
          f"| chatter: {(labels==1).sum()} "
          f"| no-chatter: {(labels==0).sum()}")

    loader = build_dataloader(windows_norm, labels, rpm_idxs)

    # ── Step 3: train ──────────────────────────────────────────────────────
    G, C = train(loader)

    # ── Step 4: generate chatter at high RPMs ─────────────────────────────
    device = next(G.parameters()).device
    synth_dfs = []
    for rpm in [7500, 8000, 8500]:
        gen_windows = generate_windows(G, label=1, rpm=rpm, n=300, device=device)
        dfs = windows_to_dataframes(gen_windows)
        for df in dfs:
            df["RPM"]   = rpm
            df["label"] = 1
        synth_dfs.extend(dfs)
        print(f"Generated 300 chatter windows at {rpm} RPM")

    # ── Step 5: sanity-check one window ───────────────────────────────────
    real_chatter_idx = np.where(labels == 1)[0][0]
    synth_sample     = generate_windows(G, label=1, rpm=4500, n=1, device=device)[0]
    plot_real_vs_synthetic(windows_norm[real_chatter_idx], synth_sample)

    # ── Step 6 (optional): extract features from synthetic windows ─────────
    # from your_feature_code import create_windows
    # all_feature_rows = []
    # for df in synth_dfs:
    #     df["RPM"] = df["RPM"].iloc[0]   # create_windows needs RPM column
    #     feats = create_windows(df, label=1)
    #     feats["rpm_group"] = df["RPM"].iloc[0]
    #     all_feature_rows.append(feats)
    # synthetic_features = pd.concat(all_feature_rows, ignore_index=True)
    # synthetic_features.to_parquet("synthetic_chatter_features.parquet", index=False)