"""
Comprehensive evaluation script for the Recursive Transformer with Diffusion.
Now synced with config.py
"""

import torch
import pandas as pd
import os
import glob
from transformers import GPT2Tokenizer

from model import TinyRecursiveTransformer, AbsorbingDiffusion
from evaluate import (
    compute_perplexity,
    evaluate_generation_quality_phi2,
    measure_diversity
)
# Import configs to ensure we match the trained model structure
from config import device, MODEL_CONFIG, DATASET_CONFIG, DIFFUSION_CONFIG, TRAINING_CONFIG

def main():
    print("="*70)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*70)

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # 1. Determine which checkpoint to load
    model_path = 'trm_diffusion_final.pt'
    
    if not os.path.exists(model_path):
        print(f"Final model {model_path} not found. Searching for step checkpoints...")
        checkpoints = glob.glob('checkpoint_step_*.pt')
        if checkpoints:
            model_path = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        else:
            print("Error: No checkpoints found!")
            return

    print(f"\nLoading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    model = TinyRecursiveTransformer(
        vocab_size=len(tokenizer) + 1,
        **MODEL_CONFIG  # Unpacks d_model, n_layers, etc. from config.py
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    diffusion = AbsorbingDiffusion(
        num_timesteps=DIFFUSION_CONFIG['num_timesteps'],
        mask_token_id=len(tokenizer)
    )

    print("Model loaded successfully!\n")

    # ===================================================================
    # METRIC 1: PERPLEXITY COMPARISON
    # ===================================================================
    print("\n" + "="*70)
    print("METRIC 1: Diffusion Loss / Perplexity")
    print("="*70)
    
    # 1. WikiText-2 (The Out-of-Distribution / Academic Standard)
    wt2_ppl = compute_perplexity(
        model, diffusion, tokenizer, 
        num_test_samples=200, 
        device=device, 
        dataset_name='wikitext',
        dataset_config='wikitext-2-raw-v1'
    )
    
    # 2. FineWeb-Edu (The In-Distribution / Training Data)
    fw_ppl = compute_perplexity(
        model, diffusion, tokenizer, 
        num_test_samples=200, 
        device=device, 
        dataset_name=DATASET_CONFIG['dataset_name'],
        dataset_config=DATASET_CONFIG.get('dataset_config', None)
    )

    # Create Comparison DataFrame
    data = {
        "Dataset": ["WikiText-2 (OOD)", "FineWeb-Edu (In-Dist)"],
        "Your Model PPL": [f"{wt2_ppl:.2f}", f"{fw_ppl:.2f}"],
        "Notes": ["Academic Benchmark", "Training Domain"]
    }
    df = pd.DataFrame(data)
    
    print("\n--- RESULTS TABLE ---")
    print(df.to_markdown(index=False))
    print("---------------------\n")


    # ===================================================================
    # METRIC 2: GENERATION QUALITY & DIVERSITY
    # ===================================================================
    print("\n" + "="*70)
    print("METRIC 2: Generation Quality")
    print("="*70)

    print("Generating samples from your model...")
    # Using recursion depth from training config or slightly deeper
    rec_depth = TRAINING_CONFIG['recursion_depth']
    
    samples = measure_diversity(
        model=model,
        diffusion=diffusion,
        tokenizer=tokenizer,
        num_samples=20, # kept small for speed
        prompt="The future of artificial intelligence",
        max_length=128,
        recursion_depth=rec_depth,
        temperature=0.9,
        device=device
    )

    print("\nEvaluating generated samples with Phi-2 (or GPT-2 Large)...")
    phi2_score, all_scores = evaluate_generation_quality_phi2(
        samples=samples,
        device=device,
        plot_histogram=True
    )

    # ===================================================================
    # SUMMARY
    # ===================================================================
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"1. WikiText-2 Score:   {wt2_ppl:.2f}")
    print(f"2. FineWeb-Edu Score:  {fw_ppl:.2f}")
    print(f"3. Generation Quality: {phi2_score:.2f} (Lower is better)")
    print("\nPlots generated:")
    print("  - generation_quality_hist.png")
    print("="*70)


if __name__ == "__main__":
    main()