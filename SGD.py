from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Optional

import torch


class SGD(torch.optim.Optimizer):
    """
    Minimal SGD optimizer with optional momentum and decoupled weight decay.

    Supports aliases:
    - alpha -> lr
    - lambda_ -> weight_decay
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        *,
        alpha: Optional[float] = None,
        lambda_: Optional[float] = None,
    ) -> None:
        if alpha is not None:
            lr = float(alpha)
        if lambda_ is not None:
            weight_decay = float(lambda_)

        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = {
            "lr": float(lr),
            "momentum": float(momentum),
            "weight_decay": float(weight_decay),
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            momentum = float(group["momentum"])
            weight_decay = float(group["weight_decay"])

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # Decoupled weight decay: p <- p - lr * wd * p
                if weight_decay != 0.0:
                    p.add_(p, alpha=-lr * weight_decay)

                # Momentum on gradient if enabled
                if momentum != 0.0:
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    buf: torch.Tensor = state["momentum_buffer"]
                    buf.mul_(momentum).add_(grad)
                    update = buf
                else:
                    update = grad

                # SGD update
                p.add_(update, alpha=-lr)

        return loss
