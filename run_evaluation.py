"""
Comprehensive evaluation script for the Recursive Transformer with Diffusion.

Computes TWO key metrics as recommended:
1. Your diffusion model's perplexity on test data (internal quality)
2. Phi-2's perplexity on your generated text (external quality)
"""

import torch
from transformers import GPT2Tokenizer

from model import TinyRecursiveTransformer, AbsorbingDiffusion
from evaluate import (
    compute_perplexity,
    evaluate_generation_quality_phi2,
    measure_diversity,
    test_diverse_prompts
)
from config import device


def main():
    print("="*70)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*70)

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    mask_token_id = len(tokenizer)
    vocab_size = len(tokenizer) + 1

    # Load trained model
    print("\nLoading trained model...")
    checkpoint = torch.load('trm_diffusion.pt', map_location=device)

    model = TinyRecursiveTransformer(
        vocab_size=vocab_size,
        d_model=516,
        n_heads=6,
        n_layers=6,
        max_seq_len=128,
        dropout=0.1
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    diffusion = AbsorbingDiffusion(
        num_timesteps=1000,
        mask_token_id=mask_token_id
    )

    print("Model loaded successfully!\n")

    # ===================================================================
    # METRIC 1: Your model's perplexity on real test data
    # ===================================================================
    print("\n" + "="*70)
    print("METRIC 1: Diffusion Model Perplexity (Reconstruction Quality)")
    print("="*70)
    print("This measures how well YOUR model can denoise/reconstruct text.")
    print()

    your_perplexity = compute_perplexity(
        model=model,
        diffusion=diffusion,
        tokenizer=tokenizer,
        num_test_samples=1000,
        device=device,
        dataset_name='openwebtext'
    )

    # ===================================================================
    # METRIC 2: Generate samples and evaluate with Phi-2
    # ===================================================================
    print("\n" + "="*70)
    print("METRIC 2: Generation Quality (Phi-2 Evaluation)")
    print("="*70)
    print("This measures how human-like/coherent your GENERATED text is.")
    print()

    print("Generating samples from your model...")
    samples = measure_diversity(
        model=model,
        diffusion=diffusion,
        tokenizer=tokenizer,
        num_samples=30,
        prompt="The future of technology",
        max_length=128,
        recursion_depth=10,
        temperature=0.9,
        device=device
    )

    print("\nEvaluating generated samples with Phi-2...")
    phi2_perplexity = evaluate_generation_quality_phi2(
        samples=samples,
        device=device
    )

    # ===================================================================
    # SUMMARY
    # ===================================================================
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"\n1. Your Model's Perplexity (on real data):  {your_perplexity:.2f}")
    print(f"   → Measures reconstruction quality")
    print(f"   → Lower = better at denoising")

    print(f"\n2. Phi-2 Perplexity (on your generations): {phi2_perplexity:.2f}")
    print(f"   → Measures generation quality")
    print(f"   → Lower = more human-like")

    print("\nInterpretation:")
    print(f"  • If YOUR perplexity is low: Model learned patterns well")
    print(f"  • If PHI-2 perplexity is low: Generations are coherent/natural")
    print(f"  • Both low = Excellent model!")
    print(f"  • Your low, Phi-2 high = Overfitting or mode collapse")
    print(f"  • Your high, Phi-2 low = Lucky samples, model needs work")

    print("\n" + "="*70)

    # Optional: Show some sample generations
    print("\nSample Generations:")
    print("="*70)
    test_diverse_prompts(
        model=model,
        diffusion=diffusion,
        tokenizer=tokenizer,
        max_length=128,
        recursion_depth=10,
        temperature=0.9,
        device=device
    )


if __name__ == "__main__":
    main()
