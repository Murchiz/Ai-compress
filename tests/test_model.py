import torch

from backend.core.model import ByteTransformer, Predictor


def test_byte_transformer_shapes():
    context_size = 128
    model = ByteTransformer(context_size=context_size)
    x = torch.randint(0, 256, (1, 10))

    # Test full forward
    logits = model(x)
    assert logits.shape == (1, 10, 256)

    # Test last_token_only
    logits_last = model(x, last_token_only=True)
    assert logits_last.shape == (1, 256)


import numpy as np


def test_predictor_prediction():
    predictor = Predictor()
    context = [10, 20, 30]
    probs = predictor.predict_next_byte_dist(context)

    assert probs.shape == (256,)
    assert np.isclose(probs.sum(), 1.0, atol=1e-5)


def test_predictor_empty_context():
    predictor = Predictor()
    probs = predictor.predict_next_byte_dist([])

    assert probs.shape == (256,)
    # Should be uniform
    assert np.allclose(probs, np.ones(256) / 256.0)
