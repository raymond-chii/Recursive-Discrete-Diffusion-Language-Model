import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        # Use batch_first=True for modern PyTorch
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
    def __init__(self, vocab_size, d_model=256, n_heads=4, n_layers=2,
                 max_seq_len=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # 1. Embeddings
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        # 2. Shared Computation Block
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        
        # 3. Stabilization for Recursion (CRITICAL FIX)
        self.ln_recycle = nn.LayerNorm(d_model)

        # 4. Readout Head
        self.ln_f = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

        # Timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )

    def timestep_embedding(self, timesteps, dim):
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

    def forward(self, x, timestep, recursion_depth=1):
        B, T = x.shape

        # Initial Embeddings
        tok_emb = self.token_embed(x)
        pos = torch.arange(T, device=x.device)
        pos_emb = self.pos_embed(pos)
        t_emb = self.time_embed(self.timestep_embedding(timestep, self.d_model))

        # Create the "Context Signal" (Input + Position + Time)
        context_signal = tok_emb + pos_emb + t_emb[:, None, :]
        
        # Initialize Hidden State
        z = self.dropout(context_signal)

        all_logits = []

        # --- THE RECURSIVE LOOP ---
        for step in range(recursion_depth):
            
            if step > 0:
                z = self.ln_recycle(z)
                
            z = z + context_signal 

            # Pass through the shared blocks
            for block in self.blocks:
                z = block(z)

            # Calculate output for this step
            out = self.ln_f(z)
            logits = self.output(out)
            all_logits.append(logits)

        # Return stack: [Depth, Batch, Time, Vocab]
        return torch.stack(all_logits)


class AbsorbingDiffusion:
    def __init__(self, num_timesteps, mask_token_id):
        self.num_timesteps = num_timesteps
        self.mask_token_id = mask_token_id

    def q_sample(self, x_0, t):
        B, T = x_0.shape

        mask_prob = t.float() / self.num_timesteps
        
        # Create mask [B, T]
        mask = torch.rand(B, T, device=x_0.device) < mask_prob[:, None]

        x_t = x_0.clone()
        x_t[mask] = self.mask_token_id
        return x_t, mask

    @torch.no_grad()
    def p_sample_step(self, model, x_t, t, recursion_depth=6, temperature=1.0):
        """Reverse step for sampling"""
        B, T = x_t.shape
        is_masked = (x_t == self.mask_token_id)
        if not is_masked.any(): return x_t

        # Get predictions
        all_logits = model(x_t, t, recursion_depth=recursion_depth)
        logits = all_logits[-1] # Use final step

        # Sample
        probs = F.softmax(logits / temperature, dim=-1)
        new_tokens = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(B, T)

        # Unmask logic
        current_ratio = t[0].float() / self.num_timesteps
        next_ratio = (t[0] - 1).float() / self.num_timesteps
        
        p_keep = next_ratio / current_ratio
        keep_mask = torch.rand(B, T, device=x_t.device) < p_keep
        
        update_locs = is_masked & (~keep_mask)
        
        x_prev = x_t.clone()
        x_prev[update_locs] = new_tokens[update_locs]
        
        return x_prev