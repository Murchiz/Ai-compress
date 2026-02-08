import torch
import time
from backend.core.model import Predictor
from backend.utils.arithmetic import ArithmeticEngine

def benchmark():
    predictor = Predictor()
    engine = ArithmeticEngine()
    context = list(range(128))

    # Warmup
    probs = predictor.predict_next_byte_dist(context)
    engine.get_cum_freqs(probs)

    n = 1000 # Increased for better measurement

    start = time.time()
    for _ in range(n):
        probs = predictor.predict_next_byte_dist(context)
        engine.get_cum_freqs(probs)
    end = time.time()
    print(f"Total time for {n} calls (Model + Engine): {end - start:.4f}s")

    start_model = time.time()
    for _ in range(n):
        probs = predictor.predict_next_byte_dist(context)
    end_model = time.time()
    print(f"Model-only time for {n} calls: {end_model - start_model:.4f}s")

    probs = predictor.predict_next_byte_dist(context)
    start_engine = time.time()
    for _ in range(n):
        engine.get_cum_freqs(probs)
    end_engine = time.time()
    print(f"Engine-only time for {n} calls: {end_engine - start_engine:.4f}s")

if __name__ == "__main__":
    benchmark()
