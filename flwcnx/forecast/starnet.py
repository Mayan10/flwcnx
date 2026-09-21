"""StarNet backbone: GRU seq2seq with periodical embedding and attention.

Reproduction of Liu et al., *Vivisecting Starlink Throughput*, Proc. ACM Netw.
3(CoNEXT4), 2025, sections 5 and 6. Not ours.

Published configuration, from CLAUDE.md section 6:

  - encoder: 2 layer GRU, hidden 128
  - decoder: 2 layer GRU, hidden 128, input size 128 + 11
  - projection head: 2 linear layers, 128 hidden, LeakyReLU, output dim 1
  - attention: three 1 layer MLPs, input size 128
  - periodical embedding: 1D conv over each of the four feature classes to
    L x 48, plus an L x 1 vector holding the second within the current 15 s
    interval, values in [0, 15)
  - AdamW, lr 1e-3, decay 0.99 per step, batch 512

Two things the paper does not pin down, marked here so the reproduction gap is
attributable if it appears:

  1. The convolution kernel width for the periodical embedding. Defaulted to 3
     with same padding, which preserves L as the shape requires.
  2. What the decoder's 11 wide input is at horizon steps beyond the first.
     Future covariates are not knowable at prediction time, so the satellite,
     time and weather channels are held at their forecast origin values and
     the throughput channel is fed the previous step's own prediction. Any
     other reading would need future information.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from flwcnx.config import FEATURE_CLASSES, PERIOD_SECONDS, StarNetConfig


class PeriodicalEmbedding(nn.Module):
    """One 1D convolution per feature class, plus the raw phase channel.

    The four classes are convolved separately rather than jointly because they
    live on unrelated scales and carry unrelated periodicity: throughput and
    satellite geometry move on the 15 s scheduling cycle, weather moves on
    hours. A single convolution over all 11 channels would force one filter
    bank to serve both.

    The phase channel is passed through unconvolved. It is the model's only
    direct view of where in the scheduling period each step sits, and the paper
    keeps it as a raw L x 1 vector in [0, 15).
    """

    def __init__(self, class_slices: dict[str, list[int]], embed_dim: int = 48,
                 kernel_size: int = 3) -> None:
        super().__init__()
        self.class_names = tuple(k for k in FEATURE_CLASSES if k in class_slices)
        self.class_slices = {k: class_slices[k] for k in self.class_names}
        self.embed_dim = embed_dim

        self.convs = nn.ModuleDict({
            name: nn.Conv1d(
                in_channels=len(self.class_slices[name]),
                out_channels=embed_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,     # same padding, so L survives
            )
            for name in self.class_names
        })
        self.activation = nn.LeakyReLU()

    @property
    def output_dim(self) -> int:
        return self.embed_dim * len(self.class_names) + 1

    def forward(self, x: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """x: (B, L, F). phase: (B, L), seconds into the period. -> (B, L, D)."""
        parts = []
        for name in self.class_names:
            channels = x[:, :, self.class_slices[name]].transpose(1, 2)  # (B, C, L)
            parts.append(self.activation(self.convs[name](channels)).transpose(1, 2))
        parts.append(phase.unsqueeze(-1))
        return torch.cat(parts, dim=-1)


class AdditiveAttention(nn.Module):
    """Three 1 layer MLPs over a 128 wide space, in the Bahdanau form.

    query projection, key projection, and the scoring projection down to a
    scalar. That is exactly three single layer MLPs of input size 128, which is
    what the paper specifies.
    """

    def __init__(self, hidden_size: int = 128) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, decoder_state: torch.Tensor,
                encoder_outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """decoder_state: (B, H). encoder_outputs: (B, L, H)."""
        energy = torch.tanh(self.query(decoder_state).unsqueeze(1) + self.key(encoder_outputs))
        weights = torch.softmax(self.score(energy).squeeze(-1), dim=1)      # (B, L)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, weights


class ProjectionHead(nn.Module):
    """2 linear layers, 128 hidden, LeakyReLU, output dim 1."""

    def __init__(self, input_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class StarNet(nn.Module):
    """The full seq2seq forecaster.

    Returns predictions in whatever space the target was given in. Training
    standardises the inputs but keeps the target in Mbps, so the output is in
    Mbps and no inverse transform is needed before the metrics.
    """

    def __init__(self, class_slices: dict[str, list[int]], horizon: int,
                 config: StarNetConfig | None = None) -> None:
        super().__init__()
        self.config = config or StarNetConfig()
        self.horizon = horizon
        self.n_features = sum(len(v) for v in class_slices.values())

        self.embedding = PeriodicalEmbedding(class_slices, self.config.embed_dim)
        self.encoder = nn.GRU(
            input_size=self.embedding.output_dim,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            batch_first=True,
            dropout=self.config.dropout if self.config.num_layers > 1 else 0.0,
        )
        # Input size 128 + 11: the attention context plus the feature vector.
        self.decoder = nn.GRU(
            input_size=self.config.hidden_size + self.n_features,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            batch_first=True,
            dropout=self.config.dropout if self.config.num_layers > 1 else 0.0,
        )
        self.attention = AdditiveAttention(self.config.hidden_size)
        self.head = ProjectionHead(self.config.hidden_size * 2, self.config.head_hidden)

    def forward(self, x: torch.Tensor, phase: torch.Tensor,
                return_attention: bool = False):
        """x: (B, L, F) standardised. phase: (B, L) in [0, 15). -> (B, horizon)."""
        embedded = self.embedding(x, phase)
        encoder_outputs, hidden = self.encoder(embedded)

        # The forecast origin: the last observed step. Future covariates are
        # not knowable, so the non-throughput channels are held here.
        origin_features = x[:, -1, :].clone()
        target_channel = 0

        outputs, attention_maps = [], []
        decoder_input_features = origin_features
        state = hidden
        query = state[-1]

        for _ in range(self.horizon):
            context, weights = self.attention(query, encoder_outputs)
            step_input = torch.cat([context, decoder_input_features], dim=-1).unsqueeze(1)
            output, state = self.decoder(step_input, state)
            query = state[-1]
            prediction = self.head(torch.cat([query, context], dim=-1))
            outputs.append(prediction)
            attention_maps.append(weights)

            # Feed the prediction back into the throughput channel only.
            decoder_input_features = decoder_input_features.clone()
            decoder_input_features[:, target_channel] = prediction.detach()

        predictions = torch.stack(outputs, dim=1)
        if return_attention:
            return predictions, torch.stack(attention_maps, dim=1)
        return predictions


class StarNetNoPeriodicalEmbedding(StarNet):
    """Ablation: the periodical embedding replaced by a plain linear projection.

    The paper reports median error rising from 33.57 to 38.00 Mbps without it.
    """

    def __init__(self, class_slices: dict[str, list[int]], horizon: int,
                 config: StarNetConfig | None = None) -> None:
        super().__init__(class_slices, horizon, config)
        n_features = self.n_features
        embed_dim = self.embedding.output_dim
        self.embedding = _LinearEmbedding(n_features, embed_dim)


class _LinearEmbedding(nn.Module):
    def __init__(self, n_features: int, output_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(n_features + 1, output_dim)
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([x, phase.unsqueeze(-1) / PERIOD_SECONDS], dim=-1))


class StarNetNoAttention(StarNet):
    """Ablation: the attention context replaced by the final encoder state.

    The paper reports median error rising from 33.57 to 37.01 Mbps without it.
    """

    def forward(self, x: torch.Tensor, phase: torch.Tensor,
                return_attention: bool = False):
        embedded = self.embedding(x, phase)
        encoder_outputs, hidden = self.encoder(embedded)
        context_fixed = encoder_outputs[:, -1, :]

        decoder_input_features = x[:, -1, :].clone()
        state = hidden
        outputs = []
        for _ in range(self.horizon):
            step_input = torch.cat([context_fixed, decoder_input_features], dim=-1).unsqueeze(1)
            _, state = self.decoder(step_input, state)
            query = state[-1]
            prediction = self.head(torch.cat([query, context_fixed], dim=-1))
            outputs.append(prediction)
            decoder_input_features = decoder_input_features.clone()
            decoder_input_features[:, 0] = prediction.detach()

        predictions = torch.stack(outputs, dim=1)
        if return_attention:
            zeros = torch.zeros(x.shape[0], self.horizon, x.shape[1], device=x.device)
            return predictions, zeros
        return predictions


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
