import numpy as np

from backend.core.model import Predictor


def test_training_sanity():
    predictor = Predictor()
    data = b"Some repetitive data to learn from. " * 50

    initial_probs = predictor.predict_next_byte_dist(list(b"Some "))

    loss = predictor.train_on_data(data, epochs=2)
    assert loss > 0

    new_probs = predictor.predict_next_byte_dist(list(b"Some "))

    # Probs should have changed
    assert not np.allclose(initial_probs, new_probs)
