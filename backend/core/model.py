import torch
import torch.nn as nn
import torch.nn.functional as F


class ByteTransformer(nn.Module):
    def __init__(
        self,
        vocab_size=256,
        embed_dim=128,
        num_layers=2,
        num_heads=4,
        context_size=128
    ):
        super().__init__()
        self.context_size = context_size
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(context_size, embed_dim)

        # Pre-calculate positional embeddings to avoid lookup overhead
        with torch.no_grad():
            pos_indices = torch.arange(context_size).unsqueeze(0)
            self.register_buffer(
                "pos_emb_cache",
                self.position_embedding(pos_indices)
            )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        self.fc_out = nn.Linear(embed_dim, vocab_size)

        # Pre-calculate positions and causal mask for performance
        self.register_buffer(
            "positions",
            torch.arange(context_size).unsqueeze(0)
        )
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(context_size, context_size),
                diagonal=1
            ).bool()
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        b, t = x.size()

        # Use pre-calculated positional embeddings and mask
        if self.training:
            pos_x = self.position_embedding(self.positions[:, :t])
        else:
            pos_x = self.pos_emb_cache[:, :t, :]

        x = self.token_embedding(x) + pos_x
        mask = self.causal_mask[:t, :t]

        x = self.transformer(x, mask=mask, is_causal=True)
        logits = self.fc_out(x)
        return logits

    def update_pos_emb_cache(self):
        """Update the positional embedding cache after training."""
        with torch.no_grad():
            self.pos_emb_cache.copy_(self.position_embedding(self.positions))


class Predictor:
    def __init__(self, model_path=None, device='cpu'):
        self.device = torch.device(device)
        self.context_size = 128
        self.model = ByteTransformer(
            context_size=self.context_size
        ).to(self.device)
        if model_path:
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device)
            )
        self.model.eval()
        self.model.update_pos_emb_cache()
        # Cache uniform distribution for empty context
        self.uniform_dist = (torch.ones(256) / 256.0).to(self.device)

    @torch.inference_mode()
    def predict_next_byte_dist(self, context_bytes):
        """
        Returns a probability distribution over 256 bytes.
        context_bytes: list or tensor of byte values.
        """
        if len(context_bytes) == 0:
            # Return cached uniform distribution
            return self.uniform_dist

        # Truncate context if too long
        if len(context_bytes) > self.context_size:
            context_bytes = context_bytes[-self.context_size:]

        x = torch.tensor(
            context_bytes,
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0)
        logits = self.model(x)
        last_logits = logits[0, -1, :]
        probs = F.softmax(last_logits, dim=-1)
        return probs

    def train_on_data(self, data_bytes, epochs=1, lr=1e-3):
        if len(data_bytes) <= 1:
            return 0

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        data_tensor = torch.tensor(
            list(data_bytes),
            dtype=torch.long,
            device=self.device
        )

        total_loss = 0
        for epoch in range(epochs):
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
