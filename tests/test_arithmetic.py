import torch

from backend.utils.arithmetic import ArithmeticEngine, Decoder, Encoder


def test_arithmetic_engine_cum_freqs():
    engine = ArithmeticEngine(precision=32)
    probs = torch.tensor([0.2, 0.3, 0.5])
    cum_freqs, total_count = engine.get_cum_freqs(probs, total_count=1000)

    assert len(cum_freqs) == 4
    assert cum_freqs[0] == 0
    assert cum_freqs[-1] == 1000
    assert total_count == 1000


def test_arithmetic_roundtrip_simple():
    engine = ArithmeticEngine(precision=32)
    # 3 symbols
    probs = torch.tensor([0.1, 0.4, 0.5])
    cum_freqs, total_count = engine.get_cum_freqs(probs)

    symbols_to_encode = [0, 1, 2, 1, 0]

    # Encode
    encoder = Encoder(engine)
    for s in symbols_to_encode:
        encoder.encode(s, cum_freqs, total_count)
    bits = encoder.finish()

    # Decode
    decoder = Decoder(engine, bits)
    decoded_symbols = []
    for _ in range(len(symbols_to_encode)):
        s = decoder.decode(cum_freqs, total_count)
        decoded_symbols.append(s)

    assert decoded_symbols == symbols_to_encode
