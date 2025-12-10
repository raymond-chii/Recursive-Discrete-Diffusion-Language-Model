import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

def validate(model, diffusion, dataloader, device, recursion_depth=8, max_steps=100):
    model.eval()
    total_loss = 0
    steps = 0
    
    with torch.no_grad():
        for i, x_0 in enumerate(tqdm(dataloader, desc="Validating", total=max_steps, leave=False)):
            if i >= max_steps:
                break
                
            x_0 = x_0.to(device)
            B = x_0.size(0)
            
            # Random timestep for diffusion loss (exclude t=0)
            t = torch.randint(1, diffusion.num_timesteps + 1, (B,), device=device)
            
            # Forward diffusion (add noise)
            x_t, mask = diffusion.q_sample(x_0, t)
            
            # Predict (denoise)
            all_logits = model(x_t, t, recursion_depth=recursion_depth)

            # Use the final (most refined) step's logits
            logits = all_logits[-1]  # Shape: [B, T, Vocab]

            # Calculate Loss (only on masked tokens)
            loss = F.cross_entropy(
                logits[mask].reshape(-1, logits.size(-1)),
                x_0[mask].reshape(-1)
            )
            
            total_loss += loss.item()
            steps += 1

    # Avoid division by zero if dataloader is empty
    return total_loss / steps if steps > 0 else 0.0


def sample(model, diffusion, tokenizer, prompt="The future of technology",
           max_length=128, recursion_depth=8, temperature=1.0, device='cuda'):
    model.eval()
    
    # Encode prompt
    prompt_tokens = tokenizer.encode(prompt, return_tensors='pt').to(device)
    prompt_len = prompt_tokens.size(1)
    
    # Initialize full sequence with mask tokens
    x_t = torch.full((1, max_length), diffusion.mask_token_id, device=device)
    
    # Paste prompt into the beginning
    x_t[0, :prompt_len] = prompt_tokens[0]

    # Iterative Denoising (Reverse Process)
    for t in tqdm(range(diffusion.num_timesteps - 1, -1, -1), desc="Sampling"):
        timestep = torch.tensor([t], device=device)
        
        # 1. Denoise one step
        x_t = diffusion.p_sample_step(
            model, x_t, timestep,
            recursion_depth=recursion_depth,
            temperature=temperature
        )
        
        # 2. Force the prompt to stay fixed 
        x_t[0, :prompt_len] = prompt_tokens[0]

    # Decode final tokens to text
    text = tokenizer.decode(x_t[0].cpu().numpy(), skip_special_tokens=True)
    return text