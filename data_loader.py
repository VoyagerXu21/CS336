# data_loader.py
from __future__ import annotations

import argparse
import os
from typing import Tuple, Union

import numpy as np
import torch


ArrayLike = Union[np.ndarray, np.memmap]


def load_tokenized_ids(
    filename: str,
    root_dir: str = "tokenized_ids",
    mmap: bool = True,
) -> ArrayLike:
    """
    Load tokenized token IDs from ./tokenized_ids/<filename> by default.

    Assumptions:
      - Saved via np.save(...) => typically .npy
      - Or saved via np.savez(...) => .npz containing one array

    Args:
        filename: file name under root_dir (e.g., "train_ids.npy")
        root_dir: directory containing tokenized ids
        mmap: whether to use memory-mapped loading when possible (recommended for large arrays)

    Returns:
        x: numpy array-like object (np.ndarray or np.memmap), integer dtype.
    """
    path = os.path.join(root_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Tokenized file not found: {path}")

    mmap_mode = "r" if mmap else None
    obj = np.load(path, mmap_mode=mmap_mode, allow_pickle=False)

    # If it's npy => obj is ndarray/memmap; if npz => obj is NpzFile
    if isinstance(obj, np.lib.npyio.NpzFile):
        keys = list(obj.keys())
        if len(keys) == 0:
            raise ValueError(f"Empty .npz file: {path}")
        if len(keys) > 1:
            raise ValueError(
                f".npz has multiple arrays {keys}. Please save a single array or specify a key in code."
            )
        x = obj[keys[0]]
    else:
        x = obj

    if not np.issubdtype(x.dtype, np.integer):
        raise TypeError(f"Expected integer token IDs, got dtype={x.dtype} from {path}")

    # Make sure we can index 1D sequence
    if x.ndim != 1:
        raise ValueError(f"Expected 1D token id sequence, got shape={x.shape} from {path}")

    return x


def get_batch(
    x: ArrayLike,
    batch_size: int,
    context_length: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a batch of training examples from a 1D token id sequence x.

    Returns:
        x_batch: (B, T) int64 tensor on device
        y_batch: (B, T) int64 tensor on device

    Where:
        x_batch[b] = x[i : i+T]
        y_batch[b] = x[i+1 : i+T+1]
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    if context_length <= 0:
        raise ValueError(f"context_length must be > 0, got {context_length}")

    n = int(x.shape[0])
    if n <= context_length:
        raise ValueError(
            f"Sequence too short: len(x)={n}, context_length={context_length} "
            f"(need at least context_length+1 tokens)"
        )

    x_t = torch.as_tensor(x)
    starts = torch.randint(0, n - context_length, (batch_size,), dtype=torch.long)
    offsets = torch.arange(context_length, dtype=torch.long).unsqueeze(0)
    idx = starts.unsqueeze(1) + offsets

    # Build batch in numpy first (works for memmap too)
    # Use pre-allocation to avoid many small arrays
    x_batch = x_t[idx]
    y_batch = x_t[idx + 1]

    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    return x_batch, y_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="filename under ./tokenized_ids (e.g., train_ids.npy)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu", help="cpu | cuda:0 | mps")
    parser.add_argument("--no_mmap", action="store_true", help="disable mmap loading")
    args = parser.parse_args()

    x = load_tokenized_ids(args.file, root_dir="tokenized_ids", mmap=(not args.no_mmap))
    xb, yb = get_batch(x, args.batch_size, args.context_length, args.device)

    print(f"Loaded: {args.file}")
    print(f"x length: {len(x)}")
    print(f"x_batch shape: {tuple(xb.shape)}, device: {xb.device}, dtype: {xb.dtype}")
    print(f"y_batch shape: {tuple(yb.shape)}, device: {yb.device}, dtype: {yb.dtype}")
    print("First row x_batch[:10]:", xb[0, :10].tolist())
    print("First row y_batch[:10]:", yb[0, :10].tolist())


if __name__ == "__main__":
    main()

