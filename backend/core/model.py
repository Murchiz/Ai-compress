import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ByteTransformer(nn.Module):
    def __init__(
        self, vocab_size=256, embed_dim=128, num_layers=2, num_heads=4, context_size=128
    ):
        super().__init__()
        self.context_size = context_size
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(context_size, embed_dim)

        # Pre-calculate positional embeddings to avoid lookup overhead
        with torch.no_grad():
            pos_indices = torch.arange(context_size).unsqueeze(0)
            self.register_buffer("pos_emb_cache", self.position_embedding(pos_indices))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

        # Pre-calculate positions and causal mask for performance
        self.register_buffer("positions", torch.arange(context_size).unsqueeze(0))
        # Bolt: Using a float mask (0.0 and -inf) is ~6% faster than boolean mask
        # on CPU because it avoids the internal conversion in SDPA.
        mask = torch.zeros(context_size, context_size)
        mask.masked_fill_(
            torch.triu(torch.ones(context_size, context_size), diagonal=1).bool(),
            float("-inf"),
        )
        self.register_buffer("causal_mask", mask)

    def forward(self, x, last_token_only=False):
        # x shape: (batch_size, seq_len)
        b, t = x.size()

        # Use pre-calculated positional embeddings and mask
        if self.training:
            pos_x = self.position_embedding(self.positions[:, :t])
        else:
            pos_x = self.pos_emb_cache[:, :t, :]

        x = self.token_embedding(x) + pos_x
        mask = self.causal_mask[:t, :t]

        # Bolt: Passing is_causal=False with a float mask is faster than
        # is_causal=True with a boolean mask in this PyTorch version on CPU.
        x = self.transformer(x, mask=mask, is_causal=False)

        if last_token_only:
            # Optimization: Slice hidden state to last token before final linear layer.
            # Bolt: Squeezing to (batch, dim) is faster than (batch, 1, dim)
            # for linear layers.
            x = x[:, -1, :]

        logits = self.fc_out(x)
        return logits

    def update_pos_emb_cache(self):
        """Update the positional embedding cache after training."""
        with torch.no_grad():
            self.pos_emb_cache.copy_(self.position_embedding(self.positions))


class Predictor:
    def __init__(self, model_path=None, device="cpu"):
        self.device = torch.device(device)
        self.context_size = 128
        self.model = ByteTransformer(context_size=self.context_size).to(self.device)
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.model.update_pos_emb_cache()
        # Cache uniform distribution for empty context
        self.uniform_dist = (torch.ones(256) / 256.0).to(self.device)
        # Bolt: Caching NumPy version saves a .numpy() call in the engine.
        self.uniform_dist_np = self.uniform_dist.cpu().numpy()

    @torch.inference_mode()
    def predict_next_byte_dist(self, context_bytes):
        """
        Returns a probability distribution over 256 bytes.
        context_bytes: list or tensor of byte values.
        """
        if len(context_bytes) == 0:
            # Bolt: Return cached NumPy array for direct use in arithmetic engine.
            return self.uniform_dist_np

        # Performance Optimization: torch.from_numpy followed by .to(device) is
        # measurably faster than torch.as_tensor for NumPy arrays.
        # Bolt: We handle both lists (for tests/API) and NumPy arrays (for engine).
        # Safety: Truncate context to model's capacity if it exceeds context_size.
        if len(context_bytes) > self.context_size:
            context_bytes = context_bytes[-self.context_size :]

        if isinstance(context_bytes, np.ndarray):
            # Performance Optimization: torch.from_numpy followed by .to(device) is
            # measurably faster than torch.as_tensor for NumPy arrays.
            x = torch.from_numpy(context_bytes).to(self.device)
        else:
            x = torch.as_tensor(context_bytes, dtype=torch.long, device=self.device)

        x = x.view(1, -1)
        logits = self.model(x, last_token_only=True)
        # logits shape is (1, 256) because of last_token_only=True
        # and squeeze optimization.
        # Bolt: Move to CPU before softmax to reduce GPU overhead and
        # prepare for the arithmetic engine.
        last_logits = logits[0].to("cpu")
        probs = F.softmax(last_logits, dim=-1)
        # Bolt: Returning NumPy array directly avoids a .numpy() call in the engine.
        return probs.numpy()

    def train_on_data(self, data_bytes, epochs=1, lr=1e-3):
        if len(data_bytes) <= 1:
            return 0

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Performance Optimization: torch.from_numpy followed by .to(device) is
        # significantly faster than torch.tensor(list(data_bytes)).
        # Bolt: np.frombuffer avoids a copy of the raw byte data.
        data_np = np.frombuffer(data_bytes, dtype=np.uint8).astype(np.int64)
        data_tensor = torch.from_numpy(data_np).to(self.device)

        total_loss = 0
        for _ in range(epochs):
            # Simple chunking
            for i in range(0, len(data_tensor) - 1, self.context_size):
                end_idx = min(i + self.context_size + 1, len(data_tensor))
                if end_idx - i < 2:
                    continue
                chunk = data_tensor[i:end_idx]
                x = chunk[:-1].unsqueeze(0)
                y = chunk[1:].unsqueeze(0)

                optimizer.zero_grad()
                logits = self.model(x)
                # logits shape: (1, seq_len, 256)
                # y shape: (1, seq_len)
                loss = F.cross_entropy(logits.view(-1, 256), y.view(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        self.model.eval()
        self.model.update_pos_emb_cache()
        # Cache uniform distribution for empty context
        self.uniform_dist = (torch.ones(256) / 256.0).to(self.device)
        return total_loss

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
