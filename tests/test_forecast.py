"""Model shapes, the published architecture constants, and the training loop."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from flwcnx.config import StarNetConfig
from flwcnx.forecast.baselines import (
    DLinear,
    XGBoostForecaster,
    build_baseline,
)
from flwcnx.forecast.starnet import (
    StarNet,
    StarNetNoAttention,
    StarNetNoPeriodicalEmbedding,
    count_parameters,
)
from flwcnx.forecast.train import train_model

CLASS_SLICES = {
    "throughput": [0],
    "satellite": [1, 2, 3, 4, 5],
    "time": [6, 7, 8, 9],
    "weather": [10, 11, 12],
}


def test_periodical_embedding_width_matches_the_paper():
    """Four feature classes convolved to L x 48, plus the raw L x 1 phase channel."""
    model = StarNet(CLASS_SLICES, horizon=5)
    assert model.embedding.output_dim == 4 * 48 + 1


def test_decoder_input_is_hidden_plus_features():
    """The paper specifies 128 + n_features."""
    model = StarNet(CLASS_SLICES, horizon=5)
    assert model.decoder.input_size == 128 + 13
    assert model.encoder.hidden_size == 128
    assert model.encoder.num_layers == 2


def test_attention_is_three_single_layer_mlps_of_width_128():
    attention = StarNet(CLASS_SLICES, horizon=5).attention
    assert attention.query.in_features == attention.query.out_features == 128
    assert attention.key.in_features == attention.key.out_features == 128
    assert attention.score.in_features == 128 and attention.score.out_features == 1


def test_embedding_preserves_sequence_length():
    model = StarNet(CLASS_SLICES, horizon=5)
    embedded = model.embedding(torch.randn(3, 30, 13), torch.rand(3, 30) * 15)
    assert embedded.shape == (3, 30, model.embedding.output_dim)


@pytest.mark.parametrize("cls", [StarNet, StarNetNoPeriodicalEmbedding, StarNetNoAttention])
def test_starnet_variants_forward_and_backward(cls):
    model = cls(CLASS_SLICES, horizon=5)
    predictions, attention = model(torch.randn(4, 30, 13), torch.rand(4, 30) * 15,
                                   return_attention=True)
    assert predictions.shape == (4, 5)
    assert attention.shape == (4, 5, 30)
    predictions.sum().backward()
    assert count_parameters(model) > 0


def test_attention_weights_are_a_distribution_over_the_lookback():
    model = StarNet(CLASS_SLICES, horizon=5)
    _, attention = model(torch.randn(4, 30, 13), torch.rand(4, 30) * 15,
                         return_attention=True)
    np.testing.assert_allclose(attention.sum(dim=-1).detach().numpy(),
                               np.ones((4, 5)), rtol=1e-5)


@pytest.mark.parametrize("name", ["dlinear", "patchtst", "timesnet"])
def test_baselines_forward_and_backward(name):
    model = build_baseline(name, lookback=30, horizon=5, n_features=13)
    predictions = model(torch.randn(4, 30, 13), torch.rand(4, 30) * 15)
    assert predictions.shape == (4, 5)
    predictions.sum().backward()


def test_unknown_baseline_is_rejected():
    with pytest.raises(ValueError, match="unknown torch baseline"):
        build_baseline("nope", lookback=30, horizon=5, n_features=13)


def test_dlinear_is_channel_independent():
    """Perturbing a non-target channel must not move the forecast."""
    model = DLinear(30, 5, target_index=0).eval()
    x = torch.randn(2, 30, 13)
    baseline = model(x)
    perturbed = x.clone()
    perturbed[:, :, 5] += 100.0
    torch.testing.assert_close(baseline, model(perturbed))


def test_xgboost_quantile_objective_tracks_the_quantile():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (3000, 5))
    y = rng.normal(0, 1, 3000)
    model = XGBoostForecaster(quantile_alpha=0.3, n_estimators=40).fit(x, y)
    # The 0.3 quantile of a standard normal is about -0.524.
    assert model.predict(x).mean() < 0.0


def test_xgboost_refuses_to_predict_before_fit():
    with pytest.raises(RuntimeError, match="before fit"):
        XGBoostForecaster().predict(np.zeros((2, 3)))


def test_training_returns_predictions_in_mbps(sequences, fitted_standardizer):
    """The model works in standardised space; the wrapper must undo it.

    The standardiser must be the one the inputs were scaled with, or the
    inverse transform is against the wrong scale and the predictions come back
    in the wrong units.
    """
    standardizer = fitted_standardizer

    train = sequences.subset(np.arange(0, 1500))
    validation = sequences.subset(np.arange(1500, 2000))
    model = StarNet(sequences.class_slices(), horizon=5)
    forecaster, history = train_model(
        model, train, validation, standardizer,
        config=StarNetConfig(epochs=2, batch_size=256), device="cpu", verbose=False,
    )
    predictions = forecaster.predict(validation)
    assert predictions.shape == validation.y.shape
    assert 50.0 < predictions.mean() < 500.0        # Mbps, not standard deviations
    assert history.epochs_run == 2
    assert forecaster.predict_horizon_mean(validation).shape == (len(validation),)
