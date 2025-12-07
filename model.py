import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.mlp(x))
        return x


class TinyRecursiveTransformer(nn.Module):
    """Tiny transformer that processes sequences recursively"""
    def __init__(self, vocab_size, d_model=256, n_heads=4, n_layers=4,
                 max_seq_len=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        # Timestep embedding for diffusion
        self.time_embed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )

        # Recursion depth embedding
        self.recursion_embed = nn.Embedding(16, d_model)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def timestep_embedding(self, timesteps, dim):
        """Sinusoidal timestep embeddings"""
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

    def forward(self, x, timestep, recursion_depth=8):
        B, T = x.shape

        # Embeddings
        tok_emb = self.token_embed(x)
        pos = torch.arange(T, device=x.device)
        pos_emb = self.pos_embed(pos)

        # Timestep conditioning
        t_emb = self.timestep_embedding(timestep, self.d_model)
        t_emb = self.time_embed(t_emb)

        # Recursion depth embedding
        rec_depth = torch.full((B,), recursion_depth, device=x.device, dtype=torch.long)
        rec_emb = self.recursion_embed(rec_depth)

        # Combine
        h = tok_emb + pos_emb + t_emb[:, None, :] + rec_emb[:, None, :]
        h = self.dropout(h)

        # Transformer
        for layer in self.layers:
            h = layer(h)

        h = self.ln_f(h)
        logits = self.output(h)
        return logits


class AbsorbingDiffusion:
    """Discrete diffusion using absorbing state (mask token)"""
    def __init__(self, num_timesteps, mask_token_id):
        self.num_timesteps = num_timesteps
        self.mask_token_id = mask_token_id

    def q_sample(self, x_0, t):
        """Forward: randomly mask tokens based on timestep"""
        B, T = x_0.shape
        mask_prob = 0.2 + 0.6 * (t.float() / self.num_timesteps)
        mask = torch.rand(B, T, device=x_0.device) < mask_prob[:, None]

        x_t = x_0.clone()
        x_t[mask] = self.mask_token_id
        return x_t, mask

    def p_sample_step(self, model, x_t, t, recursion_depth=8, temperature=1.0):
        """Reverse: unmask tokens using TRM with recursion"""
        B, T = x_t.shape
        is_masked = (x_t == self.mask_token_id)

        if not is_masked.any():
            return x_t

        # Recursive refinement
        current_x = x_t.clone()

        for rec in range(recursion_depth):
            with torch.no_grad():
                logits = model(current_x, t, recursion_depth=rec)

            probs = F.softmax(logits / temperature, dim=-1)
            new_tokens = torch.multinomial(
                probs.view(-1, probs.size(-1)), num_samples=1
            ).view(B, T)

            current_x = torch.where(is_masked, new_tokens, x_t)

        unmask_prob = 0.8 - 0.6 * (t.float() / self.num_timesteps)
        unmask = torch.rand(B, T, device=x_t.device) < unmask_prob[:, None]
        unmask = unmask & is_masked

        x_t_minus_1 = x_t.clone()
        x_t_minus_1[unmask] = current_x[unmask]
        return x_t_minus_1
