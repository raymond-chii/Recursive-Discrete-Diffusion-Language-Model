import torch
import pandas as pd
import os
import glob
from transformers import GPT2Tokenizer

from model import TinyRecursiveTransformer, AbsorbingDiffusion
from evaluate import compute_perplexity, evaluate_generation_quality_phi2, measure_diversity
from config import device, MODEL_CONFIG, DATASET_CONFIG, DIFFUSION_CONFIG, TRAINING_CONFIG

def main():
    print("="*70 + "\nCOMPREHENSIVE MODEL EVALUATION\n" + "="*70)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # 1. Find Checkpoint (Fixed Pattern)
    checkpoints = glob.glob(os.path.join('checkpoints', 'ckpt_*.pt'))
    if not checkpoints:
        print("Error: No checkpoints found in 'checkpoints/' folder!")
        return
        
    model_path = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    print(f"Loading model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)

    # Init Model
    model = TinyRecursiveTransformer(
        vocab_size=len(tokenizer) + 1,
        **MODEL_CONFIG
    ).to(device)
    
    # Load weights (Handle dictionary keys properly)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint['model_state_dict']
    model.load_state_dict(state_dict)
    model.eval()

    diffusion = AbsorbingDiffusion(
        num_timesteps=DIFFUSION_CONFIG['num_timesteps'],
        mask_token_id=len(tokenizer)
    )

    # 2. Metric 1: Perplexity
    print("\n[Metric 1] Diffusion Loss / Perplexity")
    fw_ppl = compute_perplexity(
        model, diffusion, tokenizer, 
        num_test_samples=100, device=device, 
        dataset_name=DATASET_CONFIG['dataset_name'],
        dataset_config=DATASET_CONFIG.get('dataset_config', None)
    )

    # 3. Metric 2: Generation Quality
    print("\n[Metric 2] Generation Quality")
    samples = measure_diversity(
        model, diffusion, tokenizer,
        num_samples=10, 
        prompt="The future of AI",
        max_length=128,
        recursion_depth=TRAINING_CONFIG['recursion_depth'],
        device=device
    )
    
    quality_score, _ = evaluate_generation_quality_phi2(samples, device=device)

    print("\n" + "="*70)
    print("FINAL RESULTS")
    print(f"1. FineWeb Score (Lower is better): {fw_ppl:.2f}")
    print(f"2. Gen Quality (Lower is better):   {quality_score:.2f}")
    print("="*70)

if __name__ == "__main__":
    main()