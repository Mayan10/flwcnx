"""Training loop shared by StarNet and the torch baselines.

The models predict the target in standardised space and the wrapper converts
back to Mbps. That is not just numerical hygiene: StarNet feeds its own
prediction back into the throughput channel of the decoder input, and that
channel is standardised, so predicting in the same space is what makes the
feedback consistent.

One published detail is ambiguous and is exposed as a switch rather than
guessed silently. CLAUDE.md records "decay 0.99 per step". Applied literally
per optimizer step that is a very fast decay, so `decay_per` defaults to
"step" to match what is written and can be set to "epoch" if the reproduction
comes in low and the schedule is the suspect.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from thalweg.config import StarNetConfig, resolve_device, seed_everything
from thalweg.state.features import SequenceSet, Standardizer


@dataclass
class TrainingHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    epochs_run: int = 0
    best_epoch: int = -1
    best_val_loss: float = float("inf")
    seconds: float = 0.0
    device: str = "cpu"

    def to_dict(self) -> dict:
        return {
            "train_loss": self.train_loss, "val_loss": self.val_loss,
            "epochs_run": self.epochs_run, "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss, "seconds": round(self.seconds, 1),
            "device": self.device,
        }


class PointForecaster:
    """A trained torch model plus the standardiser it was trained against.

    Holding the two together is deliberate: a prediction is only in Mbps
    because of a particular standardiser, and separating them is how a silent
    unit error gets into a results table.
    """

    def __init__(self, model: nn.Module, standardizer: Standardizer,
                 target_index: int = 0, device: str = "cpu") -> None:
        self.model = model
        self.standardizer = standardizer
        self.target_index = target_index
        self.device = device

    def predict(self, sequences: SequenceSet, batch_size: int = 1024) -> np.ndarray:
        """-> (n, horizon) in Mbps."""
        self.model.eval()
        x = torch.as_tensor(sequences.x, dtype=torch.float32)
        phase = torch.as_tensor(sequences.phase, dtype=torch.float32)

        outputs = []
        with torch.no_grad():
            for start in range(0, len(x), batch_size):
                xb = x[start : start + batch_size].to(self.device)
                pb = phase[start : start + batch_size].to(self.device)
                outputs.append(self.model(xb, pb).cpu().numpy())
        standardised = np.concatenate(outputs, axis=0)
        return self.standardizer.inverse_transform_target(standardised, self.target_index)

    def predict_horizon_mean(self, sequences: SequenceSet) -> np.ndarray:
        """One number per window, which is what an allocation decision acts on."""
        return self.predict(sequences).mean(axis=1)


def standardise_target(y: np.ndarray, standardizer: Standardizer, target_index: int) -> np.ndarray:
    mean = standardizer.mean[target_index]
    std = standardizer.std[target_index]
    return (y - mean) / std


def train_model(
    model: nn.Module,
    train: SequenceSet,
    validation: SequenceSet | None,
    standardizer: Standardizer,
    *,
    config: StarNetConfig | None = None,
    device: str = "auto",
    seed: int = 1337,
    patience: int = 10,
    decay_per: str = "step",
    verbose: bool = True,
) -> tuple[PointForecaster, TrainingHistory]:
    """Train one model and return it wrapped with its standardiser.

    Early stopping restores the best validation weights. Without that, a run
    that overfits late reports its last epoch, and the comparison between
    backbones turns into a comparison of when each one happened to stop.
    """
    config = config or StarNetConfig()
    seed_everything(seed)
    device = resolve_device(device)
    model = model.to(device)

    target_index = train.target_index
    x_train = torch.as_tensor(train.x, dtype=torch.float32)
    phase_train = torch.as_tensor(train.phase, dtype=torch.float32)
    y_train = torch.as_tensor(
        standardise_target(train.y, standardizer, target_index), dtype=torch.float32
    )

    loader = DataLoader(
        TensorDataset(x_train, phase_train, y_train),
        batch_size=config.batch_size, shuffle=True, drop_last=False,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr,
                                  weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.lr_decay)
    criterion = nn.MSELoss()

    history = TrainingHistory(device=device)
    best_state = None
    stale = 0
    started = time.time()

    for epoch in range(config.epochs):
        model.train()
        running, seen = 0.0, 0
        for xb, pb, yb in loader:
            xb, pb, yb = xb.to(device), pb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb, pb), yb)
            loss.backward()
            if config.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            if decay_per == "step":
                scheduler.step()
            running += float(loss.detach().item()) * len(xb)
            seen += len(xb)
        if decay_per == "epoch":
            scheduler.step()

        train_loss = running / max(seen, 1)
        history.train_loss.append(train_loss)

        if validation is not None and len(validation) > 0:
            val_loss = _evaluate(model, validation, standardizer, target_index, device, criterion)
            history.val_loss.append(val_loss)
            if val_loss < history.best_val_loss - 1e-6:
                history.best_val_loss = val_loss
                history.best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
        history.epochs_run = epoch + 1

        if verbose and (epoch % 5 == 0 or epoch == config.epochs - 1):
            tail = f" val {history.val_loss[-1]:.4f}" if history.val_loss else ""
            print(f"  epoch {epoch:3d}  train {train_loss:.4f}{tail}  "
                  f"lr {scheduler.get_last_lr()[0]:.2e}")

        if validation is not None and stale >= patience:
            if verbose:
                print(f"  early stop at epoch {epoch}, best was {history.best_epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    history.seconds = time.time() - started
    return PointForecaster(model, standardizer, target_index, device), history


def _evaluate(model: nn.Module, sequences: SequenceSet, standardizer: Standardizer,
              target_index: int, device: str, criterion: nn.Module,
              batch_size: int = 1024) -> float:
    model.eval()
    x = torch.as_tensor(sequences.x, dtype=torch.float32)
    phase = torch.as_tensor(sequences.phase, dtype=torch.float32)
    y = torch.as_tensor(standardise_target(sequences.y, standardizer, target_index),
                        dtype=torch.float32)
    total, seen = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = x[start : start + batch_size].to(device)
            pb = phase[start : start + batch_size].to(device)
            yb = y[start : start + batch_size].to(device)
            total += float(criterion(model(xb, pb), yb).detach().item()) * len(xb)
            seen += len(xb)
    return total / max(seen, 1)


def inference_latency_ms(model: nn.Module, sequences: SequenceSet, *, device: str = "cpu",
                         batch_size: int = 512, repeats: int = 20) -> float:
    """Median per batch inference time. StarNet report 3.9 ms per batch."""
    model = model.to(device).eval()
    x = torch.as_tensor(sequences.x[:batch_size], dtype=torch.float32).to(device)
    phase = torch.as_tensor(sequences.phase[:batch_size], dtype=torch.float32).to(device)
    timings = []
    with torch.no_grad():
        for _ in range(repeats):
            start = time.perf_counter()
            model(x, phase)
            if device == "cuda":
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000.0)
    return float(np.median(timings))
