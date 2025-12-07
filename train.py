import torch
import os
import glob
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

from model import TinyRecursiveTransformer, AbsorbingDiffusion
from dataset import TextDataset, collate_fn
from utils import train_epoch, validate, sample
from config import device, MODEL_CONFIG, DIFFUSION_CONFIG, TRAINING_CONFIG, DATASET_CONFIG, SAMPLING_CONFIG
import json


def main():
    # Check for existing checkpoints to resume from
    checkpoints = glob.glob('checkpoint_epoch_*.pt')
    start_epoch = 0

    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        print(f"\nFound checkpoint: {latest_checkpoint}")
        resume = input("Resume from this checkpoint? (y/n): ").lower().strip()

        if resume == 'y':
            print(f"Loading checkpoint...")
            checkpoint = torch.load(latest_checkpoint, map_location=device)
            start_epoch = checkpoint['epoch']
            print(f"Resuming from epoch {start_epoch}, previous loss: {checkpoint['loss']:.4f}\n")
        else:
            checkpoint = None
    else:
        checkpoint = None
    print(f"Using device: {device}")

    # Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    if checkpoint:
        vocab_size = checkpoint['vocab_size']
        mask_token_id = checkpoint['mask_token_id']
    else:
        mask_token_id = len(tokenizer)
        vocab_size = len(tokenizer) + 1

    print(f"Vocab size: {vocab_size}")
    print(f"Mask token ID: {mask_token_id}")

    # Model
    model = TinyRecursiveTransformer(
        vocab_size=vocab_size,
        d_model=MODEL_CONFIG['d_model'],
        n_heads=MODEL_CONFIG['n_heads'],
        n_layers=MODEL_CONFIG['n_layers'],
        max_seq_len=MODEL_CONFIG['max_seq_len'],
        dropout=MODEL_CONFIG['dropout']
    ).to(device)

    if checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model weights loaded from checkpoint")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params / 1e6:.2f}M")

    # Diffusion
    diffusion = AbsorbingDiffusion(
        num_timesteps=DIFFUSION_CONFIG['num_timesteps'],
        mask_token_id=mask_token_id
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAINING_CONFIG['learning_rate'])

    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Optimizer state loaded from checkpoint")

    # Load train dataset
    print("\nLoading training data...")
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        max_length=DATASET_CONFIG['max_length'],
        num_samples=DATASET_CONFIG['num_samples'],
        dataset_name=DATASET_CONFIG['dataset_name']
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=True,
        collate_fn=collate_fn
    )

    # Load validation dataset (10% of training size)
    print("\nLoading validation data...")
    val_dataset = TextDataset(
        tokenizer=tokenizer,
        max_length=DATASET_CONFIG['max_length'],
        num_samples=DATASET_CONFIG['num_samples'] // 10,
        dataset_name=DATASET_CONFIG['dataset_name']
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )

    # Track loss history for plotting
    train_losses = []  # Per-epoch average
    val_losses = []    # Per-epoch average
    all_step_losses = []  # Per-step losses
    global_step = 0

    # Training loop
    for epoch in range(start_epoch, TRAINING_CONFIG['num_epochs']):
        print(f"\n=== Epoch {epoch + 1}/{TRAINING_CONFIG['num_epochs']} ===")

        # Train
        train_loss, step_losses = train_epoch(
            model=model,
            diffusion=diffusion,
            dataloader=train_dataloader,
            optimizer=optimizer,
            device=device,
            recursion_depth=TRAINING_CONFIG['recursion_depth']
        )

        # Record step-level losses with global step counter
        for loss in step_losses:
            all_step_losses.append({'step': global_step, 'loss': loss})
            global_step += 1

        # Validate
        val_loss = validate(
            model=model,
            diffusion=diffusion,
            dataloader=val_dataloader,
            device=device,
            recursion_depth=TRAINING_CONFIG['recursion_depth']
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Steps: {global_step}")

        # Sample after each epoch
        if (epoch + 1) % 1 == 0:
            print("\n--- Sampling ---")
            text = sample(
                model=model,
                diffusion=diffusion,
                tokenizer=tokenizer,
                prompt="Once upon a time",
                max_length=SAMPLING_CONFIG['max_length'],
                recursion_depth=SAMPLING_CONFIG['recursion_depth'],
                temperature=SAMPLING_CONFIG['temperature'],
                device=device
            )
            print(f"\n{text}\n")

        # Save checkpoint after each epoch
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'vocab_size': vocab_size,
            'mask_token_id': mask_token_id,
        }
        torch.save(checkpoint, f'checkpoint_epoch_{epoch + 1}.pt')
        print(f"Checkpoint saved: checkpoint_epoch_{epoch + 1}.pt")

    # Save loss history for plotting
    loss_history = {
        'train_losses': train_losses,  # Per-epoch averages
        'val_losses': val_losses,      # Per-epoch averages
        'epochs': list(range(1, len(train_losses) + 1)),
        'step_losses': all_step_losses,  # Per-step losses for detailed plotting
        'total_steps': global_step
    }
    with open('loss_history.json', 'w') as f:
        json.dump(loss_history, f, indent=2)
    print("\nLoss history saved to loss_history.json")

    # Plot losses
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot 1: Per-epoch losses
        ax1.plot(loss_history['epochs'], train_losses, 'b-', label='Train Loss', linewidth=2, marker='o')
        ax1.plot(loss_history['epochs'], val_losses, 'r-', label='Validation Loss', linewidth=2, marker='s')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss (Per Epoch)', fontsize=14)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Per-step training loss (smoothed)
        steps = [s['step'] for s in all_step_losses]
        losses = [s['loss'] for s in all_step_losses]
        ax2.plot(steps, losses, 'b-', alpha=0.3, linewidth=0.5)
        # Moving average for smoothing
        window = min(100, len(losses) // 10)
        if window > 1:
            import numpy as np
            smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
            ax2.plot(steps[:len(smoothed)], smoothed, 'b-', linewidth=2, label=f'Smoothed (window={window})')
        ax2.set_xlabel('Training Steps', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.set_title('Training Loss (Per Step)', fontsize=14)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('loss_curves.png', dpi=300, bbox_inches='tight')
        plt.savefig('loss_curves.pdf', bbox_inches='tight')
        print("Loss curves saved: loss_curves.png, loss_curves.pdf")

        print(f"\nFinal Train Loss: {train_losses[-1]:.4f}")
        print(f"Final Val Loss: {val_losses[-1]:.4f}")
        print(f"Best Val Loss: {min(val_losses):.4f} (Epoch {val_losses.index(min(val_losses)) + 1})")
        print(f"Total Training Steps: {global_step}")

    except ImportError as e:
        print(f"matplotlib/numpy not installed, skipping plot generation: {e}")

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab_size,
        'mask_token_id': mask_token_id,
    }, 'trm_diffusion_final.pt')

    print("Final model saved!")


if __name__ == "__main__":
    main()
