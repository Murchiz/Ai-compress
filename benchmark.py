import torch
import time
from backend.utils.arithmetic import ArithmeticEngine

def benchmark_get_cum_freqs():
    engine = ArithmeticEngine()
    probs = torch.randn(256).softmax(dim=0)

    start = time.time()
    for _ in range(1000):
        engine.get_cum_freqs(probs)
    end = time.time()
    print(f"Original get_cum_freqs time for 1000 calls: {end - start:.4f}s")

class OptimizedArithmeticEngine(ArithmeticEngine):
    def get_cum_freqs(self, probs, total_count=1000000):
        probs = probs + 1e-6
        probs = probs / probs.sum()
        counts = (probs * total_count).floor().long()
        diff = total_count - counts.sum()
        counts[0] += diff

        cum_freqs = torch.zeros(257, dtype=torch.long, device=probs.device)
        cum_freqs[1:] = torch.cumsum(counts, dim=0)
        return cum_freqs.tolist(), total_count

def benchmark_optimized_get_cum_freqs():
    engine = OptimizedArithmeticEngine()
    probs = torch.randn(256).softmax(dim=0)

    start = time.time()
    for _ in range(1000):
        engine.get_cum_freqs(probs)
    end = time.time()
    print(f"Optimized get_cum_freqs time for 1000 calls: {end - start:.4f}s")

if __name__ == "__main__":
    benchmark_get_cum_freqs()
    benchmark_optimized_get_cum_freqs()
