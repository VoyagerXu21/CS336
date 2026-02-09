# adamw.py
from __future__ import annotations

from typing import Iterable, Optional, Callable, Dict, Any, Tuple

import torch
from torch.optim.optimizer import Optimizer


class AdamW(Optimizer):
    """
    AdamW optimizer (decoupled weight decay), aligned with cs336_basics.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not isinstance(betas, (tuple, list)) or len(betas) != 2:
            raise ValueError(f"betas must be a tuple/list of length 2, got {betas}")

        beta1, beta2 = float(betas[0]), float(betas[1])
        if not (0.0 <= beta1 < 1.0):
            raise ValueError(f"Invalid beta1: {beta1}")
        if not (0.0 <= beta2 < 1.0):
            raise ValueError(f"Invalid beta2: {beta2}")

        defaults: Dict[str, Any] = dict(
            lr=float(lr),
            betas=(beta1, beta2),
            eps=float(eps),
            weight_decay=float(weight_decay),
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            beta1, beta2 = group["betas"]
            eps: float = group["eps"]
            weight_decay: float = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients.")

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg: torch.Tensor = state["exp_avg"]
                exp_avg_sq: torch.Tensor = state["exp_avg_sq"]

                state["step"] += 1
                t: int = state["step"]

                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=(1.0 - beta2))

                bias_correction1 = 1.0 - (beta1**t)
                bias_correction2 = 1.0 - (beta2**t)

                step_size = lr / bias_correction1
                denom = (exp_avg_sq / bias_correction2).sqrt().add_(eps)

                if weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
