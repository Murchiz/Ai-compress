import torch
import torch.nn as nn
import time

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)
    def forward(self, x):
        return self.fc(x)

def test():
    model = SimpleModel()
    try:
        compiled_model = torch.compile(model)
        x = torch.randn(1, 10)
        # Warmup
        compiled_model(x)

        start = time.time()
        for _ in range(100):
            compiled_model(x)
        print(f"Compiled time: {time.time() - start:.4f}s")
    except Exception as e:
        print(f"Compile failed: {e}")

if __name__ == "__main__":
    test()
