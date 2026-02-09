## 2025-05-15 - [Inference and Coding Pipeline Optimization]
**Learning:** In a byte-by-byte compression loop involving both a neural network and arithmetic coding, the overhead of Python loops and redundant tensor creations (like positional embeddings and masks) can account for a significant portion of the total execution time (~30% in this case). Vectorizing the cumulative frequency calculation using `torch.cumsum` and using `register_buffer` for static tensors in the model provide measurable speedups without changing the model architecture.
**Action:** Always look for Python loops in per-byte processing paths and replace them with vectorized operations. Use `torch.inference_mode()` for a slight performance boost over `torch.no_grad()` during inference.

## 2025-05-16 - [Arithmetic Coder Loop and Search Optimization]
**Learning:** In the per-symbol loop of arithmetic coding, attribute lookups (e.g., `self.engine.MAX_RANGE`) and manual binary search in Python are significant bottlenecks. Caching these attributes as local instance variables and using the native `bisect` module for searching cumulative frequency lists can reduce the engine's overhead by over 70%.
**Action:** Cache frequently accessed engine constants in the coder's constructor. Use `bisect.bisect_right` for symbol lookup in cumulative frequency lists.

## 2025-05-17 - [Transformer Positional Embedding Optimization]
**Learning:** During per-byte inference, looking up positional embeddings via `nn.Embedding` in every forward pass is redundant. Slicing from a pre-calculated buffer (`register_buffer`) is significantly faster on CPU.
**Action:** Pre-calculate and cache positional embeddings in the model's constructor and update the cache only after training.
## 2026-02-07 - [Device Transfer Bottleneck]
**Learning:** When optimizing tensor creation from lists (e.g., using `torch.as_tensor`), always perform slicing/truncation on the host (CPU) before the transfer to the device. Moving the slice after tensor creation (`x = torch.as_tensor(list, device=device); x = x[:128]`) forces the entire list to be copied to the device, which is a major performance bottleneck for large inputs.
**Action:** Truncate input lists to the required size before calling tensor conversion functions that involve a device transfer.

## 2026-02-08 - [Redundant Linear Projection during Inference]
**Learning:** In autoregressive models like transformers, the final linear layer (projection to vocabulary) is often applied to the entire sequence. During token-by-token inference, we only need the prediction for the last token. Slicing the hidden state to `x[:, -1:, :]` before `fc_out` reduces the complexity of this layer from $O(seq\_len)$ to $O(1)$.
**Action:** Implement a `last_token_only` flag in the model's `forward` pass to skip redundant computations during inference.
