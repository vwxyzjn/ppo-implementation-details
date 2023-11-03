from __future__ import annotations
import math

import einops
import numpy as np
import torch
import torch.nn.functional as F

def support_to_value(coefficients: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    if support.ndim == 1:
        support = einops.repeat(
            support,
            "n -> b n",
            b=len(coefficients)
        )
    assert support.ndim == 2, "support must be 1 or 2 dimensional"
    assert support.shape[-1] == coefficients.shape[-1], "support and coefficients must match"
    try:
        assert torch.allclose(einops.reduce(coefficients, "... n -> ...", "sum"), torch.tensor(1.)), "coefficients must sum to 1"
    except RuntimeError:
        # happens when vmapping (currently not supported)
        ...
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
