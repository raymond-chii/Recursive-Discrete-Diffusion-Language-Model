import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


def train_epoch(model, diffusion, dataloader, optimizer, device, recursion_depth=8, log_interval=100, scheduler=None, gradient_clip=None):
    model.train()
    total_loss = 0
    step_losses = []
    total_tokens = 0
    grad_norms = []

    for step, x_0 in enumerate(tqdm(dataloader, desc="Training")):
        x_0 = x_0.to(device)
        B, T = x_0.shape

        # Count tokens in this batch
        batch_tokens = B * T
        total_tokens += batch_tokens

        # Random timesteps
        t = torch.randint(0, diffusion.num_timesteps, (B,), device=device)

        # Forward diffusion
        x_t, mask = diffusion.q_sample(x_0, t)

        # Random recursion depth
        rec_depth = torch.randint(0, recursion_depth + 1, (1,)).item()

        # Predict
        logits = model(x_t, t, recursion_depth=rec_depth)

        # Loss on masked positions
        loss = F.cross_entropy(
            logits[mask].reshape(-1, logits.size(-1)),
            x_0[mask].reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()

        # Clip gradients if specified
        if gradient_clip is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            grad_norms.append(grad_norm.item())

        optimizer.step()

        # Step learning rate scheduler
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        step_losses.append(loss.item())

    avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
    return total_loss / len(dataloader), step_losses, total_tokens, avg_grad_norm


def validate(model, diffusion, dataloader, device, recursion_depth=8):
    """Compute validation loss without updating weights"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for x_0 in tqdm(dataloader, desc="Validating"):
            x_0 = x_0.to(device)
            B = x_0.size(0)

            # Random timesteps
            t = torch.randint(0, diffusion.num_timesteps, (B,), device=device)

            # Forward diffusion
            x_t, mask = diffusion.q_sample(x_0, t)

            # Predict
            logits = model(x_t, t, recursion_depth=recursion_depth)

            # Loss on masked positions
            loss = F.cross_entropy(
                logits[mask].reshape(-1, logits.size(-1)),
                x_0[mask].reshape(-1)
            )

            total_loss += loss.item()

    return total_loss / len(dataloader)


def sample(model, diffusion, tokenizer, prompt="Once upon a time",
          max_length=128, recursion_depth=8, temperature=1.0, device='cuda'):
    model.eval()

    # Encode prompt
    prompt_tokens = tokenizer.encode(prompt, return_tensors='pt').to(device)
    prompt_len = prompt_tokens.size(1)

    # Start with fully masked (except prompt)
    x_t = torch.full((1, max_length), diffusion.mask_token_id, device=device)
    x_t[0, :prompt_len] = prompt_tokens[0]

    # Reverse diffusion
    for t in tqdm(range(diffusion.num_timesteps - 1, -1, -1), desc="Sampling"):
        timestep = torch.tensor([t], device=device)
        x_t = diffusion.p_sample_step(
            model, x_t, timestep,
            recursion_depth=recursion_depth,
            temperature=temperature
        )
        # Keep prompt fixed
        x_t[0, :prompt_len] = prompt_tokens[0]

    # Decode
    text = tokenizer.decode(x_t[0].cpu().numpy(), skip_special_tokens=True)
    return text
