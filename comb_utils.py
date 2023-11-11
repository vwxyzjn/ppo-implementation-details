from __future__ import annotations

import einops
import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "CombDistribution"
]

def support_to_value(coefficients: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    if support.ndim == 1:
        support = einops.repeat(
            support,
            "n -> b n",
            b=len(coefficients)
        )
    assert support.ndim == 2, "support must be 1 or 2 dimensional"
    assert support.shape[-1] == coefficients.shape[-1], "support and coefficients must match"
    assert torch.allclose(einops.reduce(coefficients, "... n -> ...", "sum"), torch.tensor(1.)), "coefficients must sum to 1"
    return einops.einsum(coefficients, support, "... n, ... n -> ...")

def value_to_support(values: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    assert support.dtype == values.dtype, "values and support must have the same dtype"
    if support.ndim == 1:
        support = einops.repeat(
            support,
            "n -> b n",
            b=len(values)
        )

    assert support.ndim == 2, "support must be 1 or 2 dimensional"
    assert values.ndim == 1, "values must be 1 dimensional"
    assert support.shape[-2] == values.shape[-1], "support and values must match"
    values = einops.rearrange(values, "... k -> ... k ()")
    num_values = support.shape[-1]
    upper_bounds = torch.clamp(
        torch.searchsorted(
            support,
            values,
            right=False
        ),
        1,
        num_values - 1
    )
    lower_bounds = upper_bounds - 1
    #  linear interpolate between lower and upper bound values
    interpolation = (
        values - torch.take_along_dim(support, lower_bounds, dim=-1)
    ) / (
        torch.take_along_dim(support, upper_bounds, dim=-1) - torch.take_along_dim(support, lower_bounds, dim=-1)
    )
    lower_bounds = einops.rearrange(
        lower_bounds,
        "k () -> k"
    )
    upper_bounds = einops.rearrange(
        upper_bounds,
        "k () -> k"
    )
    support = F.one_hot(
        lower_bounds,
        num_classes=num_values,
    ) * (1 - interpolation) + F.one_hot(
        upper_bounds,
        num_classes=num_values,
    ) * interpolation
    return support

class CombDistribution(torch.distributions.Distribution):
    def __init__(
        self,
        pdf: torch.Tensor,
        bounds: torch.Tensor,
    ):
        assert torch.allclose(
            einops.reduce(pdf, "... n -> ...", "sum"),
            torch.tensor(1.)
        ), "pdf must sum to 1"
        assert bounds.shape[-1] == 2, "the bounds must come in pairs (min-max)"
        batch_shape = pdf.shape[:-1]
        assert bounds.shape[:-1] == batch_shape, "the bounds must have the same batch size as the pdf"

        self.granularity = pdf.shape[-1]
        super().__init__(
            batch_shape=batch_shape,
            validate_args=False
        )

        # flatten for easier processing
        self._pdf = einops.rearrange(pdf, "... n -> (...) n")
        self._bounds = einops.rearrange(bounds, "... n -> (...) n")
        self.n = len(self._pdf)

        self._cdf = torch.concat(
            (
                torch.zeros_like(self._pdf[..., :1]),
                torch.cumsum(self._pdf, axis=-1)[:, :-2],
                torch.maximum(torch.tensor(1.), torch.sum(self._pdf, axis=-1, keepdims=True))
            ),
            axis=-1
        )
        self._points = self.generate_coefficients(bounds=self._bounds, granularity=self.granularity)
        assert self._points.shape == self._pdf.shape

    @staticmethod
    def generate_coefficients(bounds: torch.Tensor, granularity: int) -> torch.Tensor:
        assert bounds.ndim == 2, "bounds must be a 2D tensor"
        assert bounds.shape[-1], "bounds must have 2 pairs (min and max)"
        t = torch.linspace(0, 1, granularity)[:,np.newaxis].to(bounds.device)
        return torch.transpose(t * bounds[:, 1] + (1 - t) * bounds[:, 0], 0, 1)

    def rsample(self, sample_shape: torch.Size=torch.Size([])) -> torch.Tensor:
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape,)
        n = int(np.prod(sample_shape))
        # sample n points from the cdf using linear interpolation
        random_points = torch.rand(
            (self.n, n,),
            device=self._cdf.device,
        )

        coefficients = torch.stack([
            value_to_support(r, self._cdf)
            for r in torch.permute(random_points, (-1, *range(random_points.ndim - 1)))
        ])
        values = torch.stack([
            support_to_value(c, self._points)
            for c in coefficients
        ])
        return torch.reshape(values, sample_shape + self.batch_shape)

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        action_shape = action.shape
        action = torch.reshape(
            action,
            (-1, int(np.prod(self.batch_shape)))
        )

        probability_coefficients = torch.stack([
            value_to_support(a, self._points)
            for a in action
        ])

        probability_weights = support_to_value(
            probability_coefficients,
            self._pdf
        )

        return torch.log(
            torch.reshape(
                probability_weights,
                action_shape
            )
        )

    @property
    def mean(self) -> torch.Tensor:
        return torch.reshape(
            support_to_value(self._pdf, self._points),
            self.batch_shape
        )

    @property
    def variance(self) -> torch.Tensor:
        return torch.reshape(
            support_to_value(self._pdf, self._points ** 2),
            self.batch_shape
        ) - self.mean ** 2

    @property
    def mode(self) -> torch.Tensor:
        return torch.reshape(
            torch.take_along_dim(self._points, torch.argmax(self._pdf, axis=-1, keepdim=True), dim=-1),
            self.batch_shape
        )

    def entropy(self) -> torch.Tensor:
        return torch.reshape(
            -support_to_value(self._pdf, torch.log(self._pdf)),
            self.batch_shape
        )

    @property
    def kl_divergence(self, other: CombDistribution) -> torch.Tensor:
        return torch.reshape(
            -support_to_value(self._pdf, torch.log(self._pdf / other._pdf)),
            self.batch_shape
        ) * self.granularity
