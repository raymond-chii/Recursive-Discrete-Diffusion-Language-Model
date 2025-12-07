import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


def train_epoch(model, diffusion, dataloader, optimizer, device, recursion_depth=8, log_interval=100):
    model.train()
    total_loss = 0
    step_losses = []

    for step, x_0 in enumerate(tqdm(dataloader, desc="Training")):
        x_0 = x_0.to(device)
        B = x_0.size(0)

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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        step_losses.append(loss.item())

    return total_loss / len(dataloader), step_losses


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
