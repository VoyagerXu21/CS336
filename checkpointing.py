# checkpointing.py
from __future__ import annotations

import os
from typing import BinaryIO, IO, Union

import torch


OutType = Union[str, os.PathLike, BinaryIO, IO[bytes]]
SrcType = Union[str, os.PathLike, BinaryIO, IO[bytes]]


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: OutType,
) -> None:
    """
    Dump all the state from (model, optimizer, iteration) into `out`.

    Args:
        model: torch.nn.Module
        optimizer: torch.optim.Optimizer
        iteration: int
        out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
    """
    if not isinstance(iteration, int):
        raise TypeError(f"iteration must be int, got {type(iteration)}")

    ckpt = {
        "iteration": int(iteration),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    # torch.save supports both paths and file-like objects
    torch.save(ckpt, out)


def load_checkpoint(
    src: SrcType,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Load a checkpoint from `src`, restore model & optimizer states,
    and return the saved iteration number.

    Args:
        src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
        model: torch.nn.Module
        optimizer: torch.optim.Optimizer

    Returns:
        iteration: int
    """
    ckpt = torch.load(src, map_location="cpu")

    if not isinstance(ckpt, dict):
        raise TypeError(f"Checkpoint must be a dict, got {type(ckpt)}")

    # Be tolerant to possible key naming differences (but still strict enough)
    if "model_state_dict" in ckpt:
        model_sd = ckpt["model_state_dict"]
    elif "model" in ckpt:
        model_sd = ckpt["model"]
    else:
        raise KeyError("Checkpoint missing model state (expected 'model_state_dict' or 'model').")

    if "optimizer_state_dict" in ckpt:
        optim_sd = ckpt["optimizer_state_dict"]
    elif "optimizer" in ckpt:
        optim_sd = ckpt["optimizer"]
    else:
        raise KeyError(
            "Checkpoint missing optimizer state (expected 'optimizer_state_dict' or 'optimizer')."
        )

    if "iteration" not in ckpt:
        raise KeyError("Checkpoint missing 'iteration'.")

    model.load_state_dict(model_sd)
    optimizer.load_state_dict(optim_sd)

    return int(ckpt["iteration"])

