# adamw.py
from __future__ import annotations

from typing import Iterable, Optional, Callable, Dict, Any, Tuple
import math

import torch
from torch.optim.optimizer import Optimizer


class AdamW(Optimizer):
    """
    AdamW optimizer (decoupled weight decay), following Loshchilov & Hutter (2019).

    Update:
      m_t = beta1 * m_{t-1} + (1-beta1) * g_t
      v_t = beta2 * v_{t-1} + (1-beta2) * g_t^2
      alpha_t = alpha * sqrt(1-beta2^t) / (1-beta1^t)
      theta <- theta - alpha_t * m_t / (sqrt(v_t) + eps)
      theta <- theta - alpha * lambda_ * theta     (decoupled weight decay)
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        alpha: float = 1e-3,                  # learning rate (α)
        betas: Tuple[float, float] = (0.9, 0.999),  # (β1, β2)
        eps: float = 1e-8,                    # ε
        lambda_: float = 0.0,                 # weight decay (λ), decoupled
        # ---- optional aliases for compatibility ----
        lr: Optional[float] = None,           # alias of alpha
        weight_decay: Optional[float] = None  # alias of lambda_
    ) -> None:
        # ---- handle aliases ----
        if lr is not None:
            if alpha != 1e-3 and abs(alpha - lr) > 0:
                raise ValueError(f"Both alpha={alpha} and lr={lr} were provided; please use only one.")
            alpha = float(lr)

        if weight_decay is not None:
            if lambda_ != 0.0 and abs(lambda_ - weight_decay) > 0:
                raise ValueError(f"Both lambda_={lambda_} and weight_decay={weight_decay} were provided; please use only one.")
            lambda_ = float(weight_decay)

        # ---- validate hyperparams ----
        if alpha < 0.0:
            raise ValueError(f"Invalid alpha: {alpha}")
        if eps < 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if lambda_ < 0.0:
            raise ValueError(f"Invalid lambda_: {lambda_}")
        if not isinstance(betas, (tuple, list)) or len(betas) != 2:
            raise ValueError(f"betas must be a tuple/list of length 2, got {betas}")

        beta1, beta2 = float(betas[0]), float(betas[1])
        if not (0.0 <= beta1 < 1.0):
            raise ValueError(f"Invalid beta1: {beta1}")
        if not (0.0 <= beta2 < 1.0):
            raise ValueError(f"Invalid beta2: {beta2}")

        defaults: Dict[str, Any] = dict(
            alpha=float(alpha),
            betas=(beta1, beta2),
            eps=float(eps),
            lambda_=float(lambda_),
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """
        Perform one optimization step.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            alpha: float = group["alpha"]
            beta1, beta2 = group["betas"]
            eps: float = group["eps"]
            lambda_: float = group["lambda_"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients.")

                state = self.state[p]

                # ---- init state ----
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                m: torch.Tensor = state["m"]
                v: torch.Tensor = state["v"]

                # ---- update moments ----
                state["step"] += 1
                t: int = state["step"]

                # m <- beta1*m + (1-beta1)*g
                m.mul_(beta1).add_(grad, alpha=(1.0 - beta1))

                # v <- beta2*v + (1-beta2)*g^2
                v.mul_(beta2).addcmul_(grad, grad, value=(1.0 - beta2))

                # ---- bias correction ----
                bias_correction1 = 1.0 - (beta1 ** t)
                bias_correction2 = 1.0 - (beta2 ** t)
                alpha_t = alpha * math.sqrt(bias_correction2) / bias_correction1

                # ---- parameter update ----
                denom = v.sqrt().add_(eps)
                p.addcdiv_(m, denom, value=-alpha_t)

                # ---- decoupled weight decay ----
                if lambda_ != 0.0:
                    p.add_(p, alpha=-alpha * lambda_)

        return loss

