"""DDPM noise schedule.

Implements BaseNoiseSchedule with a linear beta schedule and
epsilon-prediction training objective.

Reference: Ho et al., "Denoising Diffusion Probabilistic Models" (2020).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig

from cgdap.models.base import BaseDenoiser, BaseNoiseSchedule, register_schedule


@register_schedule("ddpm")
class DDPMSchedule(BaseNoiseSchedule):
    """Linear beta DDPM with epsilon prediction.

    Args:
        train_timesteps: total diffusion steps T (default 1000)
        beta_start:      smallest noise level
        beta_end:        largest noise level
        num_train_steps: timesteps sampled per training iteration
        num_infer_steps: DDPM reverse steps during inference
    """

    def __init__(
        self,
        train_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        num_train_steps: int = 10,
        num_infer_steps: int = 100,
    ) -> None:
        super().__init__()
        self.train_timesteps = train_timesteps
        self.num_train_steps = num_train_steps
        self.num_infer_steps = num_infer_steps

        # Pre-compute schedule buffers (not learnable)
        betas = torch.linspace(beta_start, beta_end, train_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt())
        # Posterior variance for p(x_{t-1}|x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance.clamp(min=1e-20))

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "DDPMSchedule":
        d = cfg.model.ddpm
        return cls(
            train_timesteps=int(d.train_timesteps),
            beta_start=float(d.beta_start),
            beta_end=float(d.beta_end),
            num_train_steps=int(d.num_train_steps),
            num_infer_steps=int(d.num_infer_steps),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract(self, buf: torch.Tensor, t: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
        """Gather buffer values at indices t and broadcast to shape."""
        out = buf.gather(0, t)
        return out.view(t.shape[0], *([1] * (len(shape) - 1)))

    # ------------------------------------------------------------------
    # Forward process: q(x_t | x_0)
    # ------------------------------------------------------------------

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha_bar = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        x_t = sqrt_alpha_bar * x0 + sqrt_one_minus * noise
        return x_t, noise

    # ------------------------------------------------------------------
    # Inverse: reconstruct x0 from x_t and predicted noise
    # ------------------------------------------------------------------

    def predict_x0(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        pred_noise: torch.Tensor,
    ) -> torch.Tensor:
        sqrt_alpha_bar = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return (x_t - sqrt_one_minus * pred_noise) / (sqrt_alpha_bar + 1e-10)

    # ------------------------------------------------------------------
    # One DDPM reverse step: p(x_{t-1} | x_t)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def p_sample(
        self,
        denoiser: BaseDenoiser,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        betas = self._extract(self.betas, t, x_t.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        sqrt_recip_alpha = self._extract(1.0 / self.alphas.sqrt(), t, x_t.shape)

        pred_noise = denoiser(x_t, t, condition)
        model_mean = sqrt_recip_alpha * (x_t - betas / sqrt_one_minus * pred_noise)

        # Only add noise for t > 0
        posterior_var = self._extract(self.posterior_variance, t, x_t.shape)
        noise = torch.randn(
            x_t.shape,
            generator=generator,
            device=x_t.device,
            dtype=x_t.dtype,
        )
        not_last = (t > 0).float().view(-1, *([1] * (x_t.dim() - 1)))
        return model_mean + not_last * torch.sqrt(posterior_var) * noise

    # ------------------------------------------------------------------
    # Full sampling loop
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_loop(
        self,
        denoiser: BaseDenoiser,
        shape: tuple[int, ...],
        condition: torch.Tensor,
        device: torch.device,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        if generator is None and seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)

        n_steps = num_steps or self.num_infer_steps
        # Create evenly-spaced timestep schedule over [0, T-1]
        step_size = self.train_timesteps // n_steps
        timesteps = list(range(self.train_timesteps - 1, -1, -step_size))

        x = torch.randn(shape, generator=generator, device=device)
        B = shape[0]

        for t_int in timesteps:
            t = torch.full((B,), t_int, dtype=torch.long, device=device)
            x = self.p_sample(denoiser, x, t, condition, generator=generator)

        return x

    # ------------------------------------------------------------------
    # Training timestep sampler
    # ------------------------------------------------------------------

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.train_timesteps, (batch_size,), device=device)

    def extra_repr(self) -> str:
        return (
            f"train_timesteps={self.train_timesteps}, "
            f"num_train_steps={self.num_train_steps}, "
            f"num_infer_steps={self.num_infer_steps}"
        )
