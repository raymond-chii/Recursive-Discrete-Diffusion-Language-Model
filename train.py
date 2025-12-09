import torch
import os
import glob
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
import matplotlib.pyplot as plt
import numpy as np
import json
from tqdm import tqdm
import torch.nn.functional as F

from model import TinyRecursiveTransformer, AbsorbingDiffusion
from dataset import TextDataset, collate_fn
from utils import validate, sample
from config import device, MODEL_CONFIG, DIFFUSION_CONFIG, TRAINING_CONFIG, DATASET_CONFIG, SAMPLING_CONFIG

# --- CONFIG FOR CHECKPOINTS ---
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def main():
    EVAL_INTERVAL = 500 
    LOG_INTERVAL = 10 
    
    search_pattern = os.path.join(CHECKPOINT_DIR, 'checkpoint_step_*.pt')
    checkpoints = glob.glob(search_pattern)
    start_step = 0
    
    if checkpoints:
        # Sort by step number extracted from filename
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        print(f"\nFound checkpoint: {latest_checkpoint}")
        if input("Resume? (y/n): ").lower() == 'y':
            checkpoint = torch.load(latest_checkpoint, map_location=device)
            start_step = checkpoint['global_step']
            print(f"Resuming from step {start_step}...")
        else:
            checkpoint = None
    else:
        print(f"\nNo checkpoints found in '{CHECKPOINT_DIR}'. Starting fresh.")
        checkpoint = None

    # Tokenizer & Model Setup
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer) + 1
    
    model = TinyRecursiveTransformer(
        vocab_size=vocab_size,
        **MODEL_CONFIG
    ).to(device)
    
    diffusion = AbsorbingDiffusion(
        num_timesteps=DIFFUSION_CONFIG['num_timesteps'],
        mask_token_id=len(tokenizer)
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAINING_CONFIG['learning_rate'])
    
    if checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # --- DATASET (Streaming) ---
    print("\nLoading Datasets...")
    train_dataset = TextDataset(tokenizer=tokenizer, **DATASET_CONFIG)
    
    # We use a smaller validation set for speed
    val_config = DATASET_CONFIG.copy()
    val_config['num_samples'] = 10000 
    val_dataset = TextDataset(tokenizer=tokenizer, **val_config)

    train_dataloader = DataLoader(train_dataset, batch_size=TRAINING_CONFIG['batch_size'], 
                                  num_workers=4, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=TRAINING_CONFIG['batch_size'], 
                                collate_fn=collate_fn)

    # Scheduler setup
    total_steps = (DATASET_CONFIG['num_samples'] + TRAINING_CONFIG['batch_size'] - 1) // TRAINING_CONFIG['batch_size']
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=TRAINING_CONFIG['learning_rate'],
        total_steps=total_steps,
        pct_start=0.05 
    )

    # Metrics
    train_losses = []
    val_losses = []
    step_history = []
    
    # --- MAIN TRAINING LOOP ---
    model.train()
    progress_bar = tqdm(train_dataloader, total=total_steps, initial=start_step)
    
    current_loss_accum = 0
    tokens_processed = 0
    
    for step, x_0 in enumerate(progress_bar, start=start_step):
        # 1. Prepare Batch
        x_0 = x_0.to(device)
        B, T = x_0.shape
        tokens_processed += (B * T)
        
        # 2. Diffusion Process
        t = torch.randint(0, diffusion.num_timesteps, (B,), device=device)
        x_t, mask = diffusion.q_sample(x_0, t)
        
        # 3. Forward & Backward
        rec_depth = torch.randint(0, TRAINING_CONFIG['recursion_depth'] + 1, (1,)).item()
        logits = model(x_t, t, recursion_depth=rec_depth)
        
        loss = F.cross_entropy(logits[mask].reshape(-1, logits.size(-1)), x_0[mask].reshape(-1))
        
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient Clipping
        if TRAINING_CONFIG['gradient_clip']:
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING_CONFIG['gradient_clip'])
            
        optimizer.step()
        if step < total_steps:
            scheduler.step()
        
        # 4. Logging
        current_loss_accum += loss.item()
        if step % LOG_INTERVAL == 0:
            avg_loss = current_loss_accum / LOG_INTERVAL if step > 0 else current_loss_accum
            progress_bar.set_description(f"Loss: {avg_loss:.4f}")
            train_losses.append(avg_loss)
            step_history.append(step)
            current_loss_accum = 0
            
        # 5. VALIDATION INTERRUPT
        if step > 0 and step % EVAL_INTERVAL == 0:
            print(f"\n--- Validating at Step {step} ---")
            val_loss = validate(model, diffusion, val_dataloader, device, recursion_depth=TRAINING_CONFIG['recursion_depth'])
            val_losses.append({'step': step, 'loss': val_loss, 'tokens': tokens_processed})
            print(f"Validation Loss: {val_loss:.4f}")
            
            # --- CHANGED: Save Checkpoint to Folder ---
            save_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_step_{step}.pt')
            torch.save({
                'global_step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss
            }, save_path)
            print(f"Checkpoint saved to {save_path}")
            
            # Quick Sample
            print("Sampling:")
            print(sample(model, diffusion, tokenizer, device=device))
            
            model.train() 

    print("Training Complete.")
    print("\nGenerating Plots...")
    
    loss_history = {
        'train_steps': step_history,
        'train_losses': train_losses,
        'val_losses': val_losses 
    }
    with open('loss_history.json', 'w') as f:
        json.dump(loss_history, f, indent=2)

    try:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # PLOT 1: Learning Curve
        ax1 = axes[0]
        if len(train_losses) > 0:
            ax1.plot(step_history, train_losses, label='Train Loss', color='blue', alpha=0.4, linewidth=1)
            if len(train_losses) > 20:
                window = len(train_losses) // 10
                smooth_steps = step_history[window-1:]
                smooth_loss = np.convolve(train_losses, np.ones(window)/window, mode='valid')
                ax1.plot(smooth_steps, smooth_loss, label='Train Trend', color='blue', linewidth=2)
        
        if val_losses:
            val_steps = [x['step'] for x in val_losses]
            val_vals = [x['loss'] for x in val_losses]
            ax1.plot(val_steps, val_vals, 'ro-', label='Validation', markersize=5, linewidth=2)
            
        ax1.set_title('Learning Curve (Loss vs Steps)')
        ax1.set_xlabel('Global Steps')
        ax1.set_ylabel('Cross Entropy Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # PLOT 2: Scaling Law
        ax2 = axes[1]
        tokens_per_batch = TRAINING_CONFIG['batch_size'] * DATASET_CONFIG['max_length']
        train_tokens = [s * tokens_per_batch for s in step_history]
        
        if len(train_losses) > 0:
            ax2.plot(train_tokens, train_losses, color='green', alpha=0.3)

        if val_losses:
            val_toks = [x['tokens'] for x in val_losses]
            val_vals = [x['loss'] for x in val_losses]
            ax2.plot(val_toks, val_vals, 'ro-', label='Validation')

        ax2.set_title('Scaling Law (Loss vs Tokens)')
        ax2.set_xlabel('Total Tokens Processed')
        ax2.set_ylabel('Loss')
        ax2.set_xscale('log') 
        ax2.legend()
        ax2.grid(True, which="both", alpha=0.2)

        plt.tight_layout()
        plt.savefig('training_plots.png', dpi=300)
        print("Plots saved to 'training_plots.png'")

    except Exception as e:
        print(f"Plotting failed: {e}")


    final_path = os.path.join(CHECKPOINT_DIR, 'trm_diffusion_final.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': TRAINING_CONFIG
    }, final_path)
    print(f"Final model saved to {final_path}")

if __name__ == "__main__":
    main()