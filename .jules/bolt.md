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

## 2026-02-09 - [Cumulative Frequency List Construction Optimization]
**Learning:** Constructing a list of cumulative frequencies using `[0] + tensor.cumsum(0).tolist()` is significantly faster than pre-allocating a zero tensor and using in-place slice assignment (`cum_freqs[1:] = ...`). This is due to reduced overhead in Python-C++ boundary crossings and memory allocations.
**Action:** Use the `[0] + ...tolist()` pattern when converting small tensors to padded cumulative lists.

## 2026-02-11 - [Context Management and Scalar Extraction Optimization]
**Learning:** In high-frequency per-byte loops, Python list manipulations and PyTorch tensor-scalar operations have non-negligible overhead. Replacing context lists with fixed-size NumPy arrays makes `torch.as_tensor` ~7x faster. Additionally, using `.item()` to extract scalars before math operations avoids the PyTorch dispatcher overhead for 0-D tensors.
**Action:** Use NumPy arrays for sliding window contexts that are frequently converted to tensors. Always use `.item()` for scalar math in tight loops.

## 2026-02-12 - [Bit Manipulation Library Optimization]
**Learning:** Performance Insight: `bitarray` is significantly faster (~20x) than `bitstring` for both bit-level indexing and converting accumulated bits to bytes. In `bitarray` (v3.8.0), `.tobytes()` automatically zero-pads bitstreams, which is safe for this architecture since the decoder relies on the original file size header rather than the bitstream length.
**Action:** Use `bitarray` instead of `bitstring` for high-frequency bit manipulations and final serialization in compression pipelines.

## 2026-02-13 - [Tensor Casting and Adjustment Optimization]
**Learning:** In high-frequency loops, converting float tensors to longs using `p.long()` is ~2x faster than `p.floor().long()`. This is mathematically safe for positive values like scaled probabilities. Additionally, adding conditional checks (`if diff:`) to skip redundant tensor additions can provide micro-optimizations in tight Python loops.
**Action:** Use direct casting to `.long()` for positive tensors when floor behavior is needed. Avoid redundant in-place operations with simple conditional checks.

## 2026-02-14 - [Linear Layer Input Rank Optimization]
**Learning:** Passing 2D tensors `(batch, dim)` to PyTorch linear layers (`nn.Linear`) is measurably faster than passing 3D tensors `(batch, 1, dim)` during single-token inference. This reduces dispatch overhead and allows for more efficient GEMM kernels.
**Action:** Squeeze singleton dimensions (like the sequence dimension during autoregressive inference) before passing hidden states to the final linear projection layer.

## 2026-02-15 - [Inference Tensor and is_causal Bottlenecks]
**Learning:** Tensors returned from functions decorated with @torch.inference_mode() are inference tensors and do not allow in-place updates (RuntimeError). Additionally, while is_causal=True is a powerful hint for SDPA, it requires an explicit mask in some PyTorch versions/configurations when using nn.TransformerEncoder.
**Action:** Avoid in-place operations on tensors returned from inference-decorated methods. Prefer moving tensors to CPU before the final softmax to reduce GPU workload when the result is destined for a CPU-bound process like arithmetic coding.

## 2026-02-16 - [NumPy vs PyTorch for Small Symbol Sets]
**Learning:** For small-scale mathematical operations (e.g., probability-to-frequency conversion for 256 symbols), NumPy is significantly faster (>2x) than PyTorch. This is because PyTorch's dispatcher and kernel launch overhead are relatively high compared to the actual compute for small arrays on the CPU.
**Action:** Use NumPy for post-processing model outputs (like probability distribution adjustments) before they enter CPU-bound logic like arithmetic coding.

## 2026-02-17 - [Host-Side Truncation and Tensor Creation Overhead]
**Learning:** Moving truncation logic to after a device transfer is a major performance regression as it forces unnecessary data to be moved across the bus. Truncating on the host (CPU) first is critical. Additionally, while `torch.as_tensor` is versatile, `torch.from_numpy().to(device)` is approximately 40% faster for high-frequency conversion of existing NumPy arrays.
**Action:** Always truncate input data on the CPU before converting to tensors. Use `torch.from_numpy().to(device)` for maximum efficiency when the input is a compatible NumPy array.
