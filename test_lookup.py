import time
import torch
import numpy as np

def test_lookup():
    size = 257
    l = list(range(size))
    n = np.array(l)
    t = torch.tensor(l)

    iters = 100000

    start = time.time()
    for i in range(iters):
        _ = l[i % size]
    print(f"List lookup: {time.time() - start:.4f}s")

    start = time.time()
    for i in range(iters):
        _ = n[i % size]
    print(f"Numpy lookup: {time.time() - start:.4f}s")

    start = time.time()
    for i in range(iters):
        _ = t[i % size].item()
    print(f"Tensor lookup: {time.time() - start:.4f}s")

if __name__ == "__main__":
    test_lookup()
