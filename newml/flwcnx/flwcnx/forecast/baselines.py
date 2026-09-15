"""Baseline forecasters.

All torch models take the same `(x, phase)` signature as StarNet so the
training loop does not have to special case them. `phase` is ignored by the
generic backbones, which is the point: none of them has a notion of a
scheduling period, and whether that costs them is part of what the comparison
measures.

  - DLinear     Zeng et al., AAAI 2023
  - PatchTST    Nie et al., arXiv 2211.14730
  - TimesNet    Wu et al., arXiv 2210.02186
  - XGBoost     Chen and Guestrin, KDD 2016. Stands in for T3P, which uses
                gradient boosted trees over a flattened window, and is also the
                backbone BG-CFQS trains with pinball loss.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SeriesDecomposition(nn.Module):
    """Moving average trend plus seasonal residual (DLinear section 3.2)."""

    def __init__(self, kernel_size: int = 25) -> None:
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, L, C) -> (seasonal, trend)."""
        pad = self.kernel_size // 2
        # Replicate padding at both ends, so the trend does not get dragged
        # toward zero at the window edges.
        padded = F.pad(x.transpose(1, 2), (pad, pad), mode="replicate")
        trend = F.avg_pool1d(padded, kernel_size=self.kernel_size, stride=1)
        trend = trend[:, :, : x.shape[1]].transpose(1, 2)
        return x - trend, trend


class DLinear(nn.Module):
    """Decomposition plus two linear maps from look-back to horizon.

    Channel independent, as published: the target channel is forecast from its
    own history. That is the whole argument of the paper, so weakening it into
    a multivariate model would not be the baseline it claims to be.
    """

    def __init__(self, lookback: int, horizon: int, target_index: int = 0,
                 kernel_size: int = 25) -> None:
        super().__init__()
        self.target_index = target_index
        self.decomposition = SeriesDecomposition(kernel_size)
        self.seasonal = nn.Linear(lookback, horizon)
        self.trend = nn.Linear(lookback, horizon)

    def forward(self, x: torch.Tensor, phase: torch.Tensor | None = None) -> torch.Tensor:
        series = x[:, :, self.target_index : self.target_index + 1]
        seasonal, trend = self.decomposition(series)
        out = (self.seasonal(seasonal.squeeze(-1).unsqueeze(1))
               + self.trend(trend.squeeze(-1).unsqueeze(1)))
        return out.squeeze(1)


class PatchTST(nn.Module):
    """Patching plus a transformer encoder plus a flattened linear head.

    Channel independence and patching are the two ideas: each channel is
    embedded on its own, and the sequence the transformer sees is patches
    rather than time steps, which cuts the attention cost and gives each token
    a local context wider than one sample.
    """

    def __init__(self, lookback: int, horizon: int, n_features: int,
                 patch_length: int = 8, stride: int = 4, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1,
                 target_index: int = 0) -> None:
        super().__init__()
        self.patch_length = patch_length
        self.stride = stride
        self.target_index = target_index
        self.n_patches = max(1, (lookback - patch_length) // stride + 1)

        self.embed = nn.Linear(patch_length, d_model)
        self.position = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(self.n_patches * d_model, horizon)

    def forward(self, x: torch.Tensor, phase: torch.Tensor | None = None) -> torch.Tensor:
        series = x[:, :, self.target_index]                       # (B, L)
        patches = series.unfold(dimension=1, size=self.patch_length, step=self.stride)
        patches = patches[:, : self.n_patches, :]                 # (B, P, patch_length)

        # RevIN in its simplest form: per window normalisation, undone at the
        # output. Without it the model has to learn the offset of every window.
        mean = patches.mean(dim=(1, 2), keepdim=True)
        std = patches.std(dim=(1, 2), keepdim=True).clamp_min(1e-5)
        tokens = self.embed((patches - mean) / std) + self.position
        encoded = self.encoder(tokens).flatten(1)
        return self.head(encoded) * std.squeeze(-1) + mean.squeeze(-1)


class InceptionBlock(nn.Module):
    """Multi kernel 2D convolution, the TimesBlock's inner operator."""

    def __init__(self, in_channels: int, out_channels: int, n_kernels: int = 4) -> None:
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i)
            for i in range(n_kernels)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([branch(x) for branch in self.branches], dim=0).mean(dim=0)


class TimesBlock(nn.Module):
    """Fold the series on its dominant periods and convolve in 2D.

    The idea: a 1D series with period p reshaped to (p, L/p) puts intra-period
    variation along one axis and inter-period variation along the other, so a
    2D convolution sees both at once. Periods come from the FFT amplitude
    spectrum, and the k reshapes are recombined weighted by their amplitude.
    """

    def __init__(self, seq_len: int, d_model: int, d_ff: int = 32, top_k: int = 3) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.top_k = top_k
        self.conv = nn.Sequential(
            InceptionBlock(d_model, d_ff),
            nn.GELU(),
            InceptionBlock(d_ff, d_model),
        )

    @staticmethod
    def _periods(x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        spectrum = torch.fft.rfft(x, dim=1)
        amplitude = spectrum.abs().mean(dim=0).mean(dim=-1)
        amplitude[0] = 0.0                     # the mean is not a period
        k = min(k, max(1, amplitude.shape[0] - 1))
        _, top = torch.topk(amplitude, k)
        periods = torch.clamp(x.shape[1] // top.clamp(min=1), min=1)
        weights = spectrum.abs().mean(dim=-1)[:, top]
        return periods, weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, channels = x.shape
        periods, weights = self._periods(x, self.top_k)

        folded = []
        for period in periods.tolist():
            period = max(1, int(period))
            padded_length = int(np.ceil(length / period) * period)
            if padded_length > length:
                pad = torch.zeros(batch, padded_length - length, channels, device=x.device,
                                  dtype=x.dtype)
                series = torch.cat([x, pad], dim=1)
            else:
                series = x
            grid = series.reshape(batch, padded_length // period, period, channels)
            grid = grid.permute(0, 3, 1, 2).contiguous()      # (B, C, rows, period)
            out = self.conv(grid).permute(0, 2, 3, 1).reshape(batch, padded_length, channels)
            folded.append(out[:, :length, :])

        stacked = torch.stack(folded, dim=-1)                 # (B, L, C, k)
        weights = torch.softmax(weights, dim=1).unsqueeze(1).unsqueeze(1)
        return torch.sum(stacked * weights, dim=-1) + x       # residual


class TimesNet(nn.Module):
    """A stack of TimesBlocks with a linear predictor on top."""

    def __init__(self, lookback: int, horizon: int, n_features: int,
                 d_model: int = 32, d_ff: int = 32, n_layers: int = 2,
                 top_k: int = 3, target_index: int = 0) -> None:
        super().__init__()
        self.target_index = target_index
        self.embed = nn.Linear(n_features, d_model)
        self.blocks = nn.ModuleList([
            TimesBlock(lookback, d_model, d_ff, top_k) for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.head = nn.Linear(lookback * d_model, horizon)

    def forward(self, x: torch.Tensor, phase: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.embed(x)
        for block, norm in zip(self.blocks, self.norms, strict=True):
            hidden = norm(block(hidden))
        return self.head(hidden.flatten(1))


class XGBoostForecaster:
    """Gradient boosted trees over a flattened look-back window.

    Stands in for T3P and is the BG-CFQS backbone. With `quantile_alpha` set it
    trains against the pinball loss, which is what the budget guided quantile
    selection needs: one model per candidate quantile is prohibitive, so
    BG-CFQS trains a small set and searches over them.
    """

    def __init__(self, *, quantile_alpha: float | None = None, n_estimators: int = 300,
                 max_depth: int = 6, learning_rate: float = 0.05, seed: int = 1337,
                 n_jobs: int = -1) -> None:
        self.quantile_alpha = quantile_alpha
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "random_state": seed,
            "n_jobs": n_jobs,
            "tree_method": "hist",
        }
        self._model = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> XGBoostForecaster:
        import xgboost as xgb

        params = dict(self.params)
        if self.quantile_alpha is not None:
            params["objective"] = "reg:quantileerror"
            params["quantile_alpha"] = float(self.quantile_alpha)
        else:
            params["objective"] = "reg:squarederror"
        self._model = xgb.XGBRegressor(**params)
        self._model.fit(x, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("XGBoostForecaster.predict before fit")
        return np.asarray(self._model.predict(x), dtype=float).reshape(len(x))

    @property
    def feature_importance(self) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("no fitted model")
        return np.asarray(self._model.feature_importances_, dtype=float)


def build_baseline(name: str, *, lookback: int, horizon: int, n_features: int,
                   target_index: int = 0, **kwargs):
    """Factory so the runner can name a baseline in a config."""
    name = name.lower()
    if name == "dlinear":
        return DLinear(lookback, horizon, target_index, **kwargs)
    if name == "patchtst":
        return PatchTST(lookback, horizon, n_features, target_index=target_index, **kwargs)
    if name == "timesnet":
        return TimesNet(lookback, horizon, n_features, target_index=target_index, **kwargs)
    raise ValueError(
        f"unknown torch baseline {name!r}; XGBoost is not a torch model, "
        "build XGBoostForecaster directly"
    )
