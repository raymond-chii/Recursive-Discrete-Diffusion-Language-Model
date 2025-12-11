import torch
import os
import glob
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
import numpy as np
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
from torch.amp import autocast, GradScaler 

from model import TinyRecursiveTransformer, AbsorbingDiffusion
from dataset import TextDataset, collate_fn
from config import device, MODEL_CONFIG, DIFFUSION_CONFIG, TRAINING_CONFIG, DATASET_CONFIG

# --- A100 OPTIMIZATIONS ---
if TRAINING_CONFIG.get('use_tf32', True):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def validate(model, diffusion, dataloader, tokenizer, device, recursion_depth):
    model.eval()
    easy_loss, hard_loss = 0, 0
    easy_count, hard_count = 0, 0
    dtype = torch.bfloat16 if TRAINING_CONFIG.get('mixed_precision') == 'bf16' else torch.float16

    with torch.no_grad():
        for i, x_0 in enumerate(dataloader):
            if i >= 50: break 
            x_0 = x_0.to(device)
            B = x_0.shape[0]
            
            t = torch.randint(1, diffusion.num_timesteps + 1, (B,), device=device)
            x_t, mask = diffusion.q_sample(x_0, t)
            
            with autocast('cuda', dtype=dtype):
                all_logits = model(x_t, t, recursion_depth=recursion_depth)
                final_logits = all_logits[-1]
                
                for b in range(B):
                    t_val = t[b].item()
                    m_idx = mask[b]
                    if m_idx.sum() > 0:
                        loss_val = F.cross_entropy(final_logits[b][m_idx], x_0[b][m_idx]).item()
                        if t_val <= 5: easy_loss += loss_val; easy_count += 1
                        elif t_val > 40: hard_loss += loss_val; hard_count += 1

    avg_easy = easy_loss / max(1, easy_count)
    avg_hard = hard_loss / max(1, hard_count)
    print(f"\n[METRICS] Easy Loss: {avg_easy:.4f} | Hard Loss: {avg_hard:.4f}")

    print("\n--- SAMPLING DEMO ---")
    sample_B = 2
    x_t = torch.full((sample_B, 64), diffusion.mask_token_id, dtype=torch.long, device=device)
    for i in range(diffusion.num_timesteps, 0, -1):
        t_tensor = torch.full((sample_B,), i, device=device, dtype=torch.long)
        x_t = diffusion.p_sample_step(model, x_t, t_tensor, recursion_depth=recursion_depth)
        if i % 10 == 0: print(f".", end="", flush=True)
    print()
    for idx in range(sample_B):
        text = tokenizer.decode(x_t[idx], skip_special_tokens=True)
        print(f"Sample {idx+1}: {text}")
        print("-" * 40)
        
    model.train()
    return (avg_easy + avg_hard) / 2

def generate_plots(history):
    steps = history['train_steps']
    losses = history['train_losses']
    val_steps = [x['step'] for x in history['val_losses']]
    val_losses = [x['loss'] for x in history['val_losses']]

    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, label='Training Loss', color='#1f77b4', alpha=0.6)
    if len(losses) > 10:
        window = min(50, len(losses) // 5)
        if window > 0:
            smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
            plt.plot(steps[window-1:], smoothed, label='Trend', color='#000080', linewidth=2)
    if val_losses:
        plt.plot(val_steps, val_losses, 'o-', label='Validation', color='#d62728', linewidth=2)

    plt.title('Recursive Diffusion Training Dynamics (A100/BF16)')
    plt.xlabel('Global Steps')
    plt.ylabel('Weighted Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('loss_plot.png', dpi=300)

def main():
    ACCUM_STEPS = TRAINING_CONFIG.get('accum_steps', 1)
    LOG_INTERVAL = 50 
    EVAL_INTERVAL = 1000 
    
    # 1. Setup
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer) + 1
    
    model = TinyRecursiveTransformer(vocab_size=vocab_size, **MODEL_CONFIG).to(device)
    diffusion = AbsorbingDiffusion(num_timesteps=DIFFUSION_CONFIG['num_timesteps'], mask_token_id=len(tokenizer))
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAINING_CONFIG['learning_rate'])
    scaler = GradScaler('cuda', enabled=False) 

    # 2. Checkpoint Loading
    start_step = 0
    checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, 'ckpt_*.pt'))
    if checkpoints:
        latest = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        print(f"Resuming from {latest}")
        ckpt = torch.load(latest, map_location=device)
        
        state_dict = ckpt['model']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('_orig_mod.'):
                new_state_dict[k[10:]] = v # Remove first 10 chars
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict)
        # --------------------------------------

        optimizer.load_state_dict(ckpt['opt'])
        start_step = ckpt['global_step']
        print(f"Resumed at step {start_step}")

    # --- TORCH.COMPILE ---
    print("Compiling model... (Hang tight for ~60s)")
    model = torch.compile(model)

    # 3. Dataset
    print("Loading Dataset (Slice & Cache)...")
    train_dataset = TextDataset(tokenizer=tokenizer, **DATASET_CONFIG)
    val_config = DATASET_CONFIG.copy(); val_config['num_samples'] = 2000
    val_dataset = TextDataset(tokenizer=tokenizer, **val_config)

    print("Using 8 Data Workers for max speed.")
    train_dataloader = DataLoader(train_dataset, batch_size=TRAINING_CONFIG['batch_size'], num_workers=8, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=TRAINING_CONFIG['batch_size'], collate_fn=collate_fn)

    # 4. Scheduler
    total_steps = (DATASET_CONFIG['num_samples'] // TRAINING_CONFIG['batch_size']) * TRAINING_CONFIG['num_epochs']
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=TRAINING_CONFIG['learning_rate'], total_steps=total_steps, pct_start=0.05)
    
    if start_step > 0:
        for _ in range(start_step): scheduler.step()

    print(f"Starting Training: {TRAINING_CONFIG['num_epochs']} Epochs, {total_steps} Steps.")

    # 5. Loop & History Logic
    loss_history = {'train_steps': [], 'train_losses': [], 'val_losses': []}
    
    if os.path.exists('loss_history.json') and start_step > 0:
        print("Loaded previous training history.")
        with open('loss_history.json', 'r') as f: loss_history = json.load(f)

    model.train()
    pbar = tqdm(total=total_steps, initial=start_step, desc="Training")
    global_step = start_step
    current_loss_accum = 0
    dtype = torch.bfloat16 
    
    for epoch in range(TRAINING_CONFIG['num_epochs']):
        for batch_idx, x_0 in enumerate(train_dataloader):
            x_0 = x_0.to(device)
            B, T = x_0.shape
            
            t = torch.randint(1, diffusion.num_timesteps + 1, (B,), device=device)
            x_t, mask = diffusion.q_sample(x_0, t)
            max_d = TRAINING_CONFIG['recursion_depth']
            current_depth = torch.randint(1, max_d + 1, (1,)).item()

            with autocast('cuda', dtype=dtype):
                all_logits = model(x_t, t, recursion_depth=current_depth)
                p_mask = t.float() / diffusion.num_timesteps
                p_mask = p_mask.clamp(min=0.25) 
                
                total_loss = 0
                for i in range(current_depth):
                    loss_nll = F.cross_entropy(all_logits[i].transpose(1,2), x_0, reduction='none')
                    loss_nll = (loss_nll * mask.float()).sum() / mask.sum().clamp(min=1.0)
                    total_loss += (loss_nll / p_mask.mean())
                
                final_loss = total_loss / current_depth
                final_loss = final_loss / ACCUM_STEPS

            final_loss.backward()
            
            if (batch_idx + 1) % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                
                global_step += 1
                pbar.update(1)
                
                current_loss_accum += (final_loss.item() * ACCUM_STEPS)
                
                if global_step % LOG_INTERVAL == 0:
                    avg_loss = current_loss_accum / LOG_INTERVAL
                    pbar.set_description(f"Loss: {avg_loss:.4f} | D:{current_depth}")
                    loss_history['train_steps'].append(global_step)
                    loss_history['train_losses'].append(avg_loss)
                    current_loss_accum = 0

                if global_step % EVAL_INTERVAL == 0:
                    print(f"\n--- Validation Step {global_step} ---")
                    val_loss = validate(model, diffusion, val_dataloader, tokenizer, device, max_d)
                    loss_history['val_losses'].append({'step': global_step, 'loss': val_loss})
                    
                    save_path = os.path.join(CHECKPOINT_DIR, f'ckpt_{global_step}.pt')
                    torch.save({
                        'global_step': global_step, 'model': model.state_dict(), 'opt': optimizer.state_dict()
                    }, save_path)
                    
                    # Auto-delete
                    all_ckpts = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, 'ckpt_*.pt')), key=lambda x: int(x.split('_')[-1].split('.')[0]))
                    if len(all_ckpts) > 3:
                        for old_ckpt in all_ckpts[:-3]:
                            try: os.remove(old_ckpt); print(f"Deleted old checkpoint: {old_ckpt}")
                            except OSError: pass

                    with open('loss_history.json', 'w') as f: json.dump(loss_history, f, indent=2)
                    generate_plots(loss_history)

    print("Done.")
    generate_plots(loss_history)

if __name__ == "__main__":
    main()