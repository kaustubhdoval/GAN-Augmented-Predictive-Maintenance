"""
generate_from_checkpoint.py
────────────────────────────
Load a saved .pt checkpoint and generate synthetic chatter windows
without importing the full training script.

Usage:
    python generate_from_checkpoint.py \
        --checkpoint GeneratingFailureData/checkpoints/checkpoint_epoch_00500.pt \
        --output     dataset/synthetic \
        --n          300
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from GeneratingFailureData.gan_model import Generator, generate_windows, save_windows_to_csv

# ── Copy of the architecture constants ──────────────────
WINDOW_SIZE    = 200
N_CHANNELS     = 3
NOISE_DIM      = 128
GEN_BATCH_SIZE = 256

RPM_CLASSES    = [4500, 5500, 6000, 7500, 8000, 8500]
NUM_RPM        = len(RPM_CLASSES)
NUM_LABELS     = 2
COND_DIM       = NUM_LABELS + NUM_RPM   # 8

CHANNELS_G     = [256, 128, 64, 32]

rpm_to_idx     = {rpm: i for i, rpm in enumerate(RPM_CLASSES)}


# ── CLI entry point ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic chatter windows from a saved checkpoint.")
    p.add_argument("--checkpoint", required=True,
                   help="Path to .pt checkpoint file")
    p.add_argument("--output",     default="dataset/synthetic",
                   help="Output directory for CSV files (default: dataset/synthetic)")
    p.add_argument("--n",          type=int, default=300,
                   help="Windows to generate per RPM/label combination (default: 300)")
    p.add_argument("--rpms",       type=int, nargs="+", default=[7500, 8000, 8500],
                   help="RPM values to generate for (default: 7500 8000 8500)")
    p.add_argument("--label",      type=int, default=1, choices=[0, 1],
                   help="Label to generate: 1=chatter, 0=no-chatter (default: 1)")
    p.add_argument("--device",     default="auto",
                   help="'auto', 'cpu', or 'cuda' (default: auto)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Device ────────────────────────────────────────────────────────────
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # ── Load checkpoint ───────────────────────────────────────────────────
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)

    G = Generator().to(device)
    G.load_state_dict(ckpt["G_state"])
    print(f"Loaded generator from epoch {ckpt['epoch']}")

    # ── Validate RPMs ─────────────────────────────────────────────────────
    for rpm in args.rpms:
        if rpm not in rpm_to_idx:
            raise ValueError(f"RPM {rpm} not in training classes {RPM_CLASSES}. "
                             f"Re-train with this RPM or choose from {RPM_CLASSES}.")

    # ── Generate ──────────────────────────────────────────────────────────
    all_windows, all_labels, all_rpms = [], [], []

    for rpm in args.rpms:
        print(f"Generating {args.n} windows — label={args.label}, RPM={rpm} ...", end=" ")
        wins = generate_windows(G, label=args.label, rpm=rpm, n=args.n, device=device)
        all_windows.append(wins)
        all_labels.append(np.full(args.n, args.label, dtype=np.int64))
        all_rpms.append(np.full(args.n, rpm,         dtype=np.int64))
        print("done")

    all_windows = np.concatenate(all_windows, axis=0)
    all_labels  = np.concatenate(all_labels,  axis=0)
    all_rpms    = np.concatenate(all_rpms,    axis=0)

    # ── Save ──────────────────────────────────────────────────────────────
    save_windows_to_csv(all_windows, all_labels, all_rpms, args.output)
    print(f"\nTotal: {len(all_windows)} windows written to {args.output}/")


if __name__ == "__main__":
    main()