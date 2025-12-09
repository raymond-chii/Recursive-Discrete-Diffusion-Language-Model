import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from datasets import load_dataset
from utils import sample
from tqdm import tqdm

def measure_diversity(model, diffusion, tokenizer, num_samples=30, prompt="Once upon a time",
                      max_length=64, recursion_depth=10, temperature=0.9, device='cuda'):
    print(f"Generating {num_samples} samples to measure diversity...\n")

    samples = []
    for i in range(num_samples):
        text = sample(
            model=model,
            diffusion=diffusion,
            tokenizer=tokenizer,
            prompt=prompt,
            max_length=max_length,
            recursion_depth=recursion_depth,
            temperature=temperature,
            device=device
        )
        samples.append(text)

    # Compute diversity
    all_tokens = []
    for text in samples:
        tokens = text.lower().split()
        all_tokens.extend(tokens)

    unique_tokens = len(set(all_tokens))
    total_tokens = len(all_tokens)
    ttr = unique_tokens / total_tokens if total_tokens > 0 else 0

    print(f"\nDiversity Metrics:")
    print(f"  Unique tokens: {unique_tokens} / {total_tokens}")
    print(f"  Type-Token Ratio: {ttr:.3f}")

    return samples


def compute_perplexity(model, diffusion, tokenizer, num_test_samples=1000, device='cuda', dataset_name=None, dataset_config=None):
    """
    Computes diffusion loss (proxy for perplexity) on a held-out dataset.
    Handles streaming datasets robustly.
    """
    print(f"Loading test data ({dataset_name})...")
    
    try:
        if dataset_name == 'wikitext':
            test_dataset = load_dataset(dataset_name, dataset_config, split='test', streaming=True)
        elif 'fineweb' in dataset_name.lower():
            test_dataset = load_dataset(dataset_name, dataset_config, split='train', streaming=True)
            test_dataset = test_dataset.skip(12000000)
        else:
            test_dataset = load_dataset(dataset_name, split='train', streaming=True)
            test_dataset = test_dataset.skip(1000000)
    except Exception as e:
        print(f"Warning: Could not load {dataset_name} ({e}). using OpenWebText fallback.")
        test_dataset = load_dataset('openwebtext', split='train', streaming=True).skip(1000)

    test_samples = []
    # robust iterator
    iterator = iter(test_dataset)
    for _ in range(num_test_samples):
        try:
            example = next(iterator)
            test_samples.append(example['text'])
        except StopIteration:
            break

    print(f"Loaded {len(test_samples)} test samples")

    model.eval()
    total_loss = 0
    total_tokens = 0
    fixed_timestep = diffusion.num_timesteps // 2

    with torch.no_grad():
        for text in tqdm(test_samples, desc=f"Eval {dataset_name}"):
            if not text.strip(): continue
            
            # Truncate to model max length
            tokens = tokenizer.encode(text, return_tensors='pt',
                                     truncation=True, max_length=128).to(device)

            if tokens.size(1) < 5:
                continue

            # FIXED timestep for consistent evaluation
            t = torch.tensor([fixed_timestep], device=device)

            # Add noise
            x_t, mask = diffusion.q_sample(tokens, t)

            if not mask.any():
                continue

            # Predict
            logits = model(x_t, t, recursion_depth=4) # Use moderate depth for eval

            # Loss
            loss = F.cross_entropy(
                logits[mask].reshape(-1, logits.size(-1)),
                tokens[mask].reshape(-1),
                reduction='sum'
            )
            total_loss += loss.item()
            total_tokens += mask.sum().item()

    # Perplexity = exp(average NLL)
    perplexity = np.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')

    print(f"\n=== Diffusion Model Perplexity (t={fixed_timestep}/{diffusion.num_timesteps}) ===")
    print(f"  Dataset: {dataset_name}")
    print(f"  Perplexity: {perplexity:.2f}")

    return perplexity


def evaluate_generation_quality_phi2(samples, device='cuda', plot_histogram=True):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\nLoading Phi-2 for quality evaluation...")

    try:
        eval_model_name = 'microsoft/phi-2'
        phi2_model = AutoModelForCausalLM.from_pretrained(
            eval_model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).to(device)
        phi2_tokenizer = AutoTokenizer.from_pretrained(eval_model_name, trust_remote_code=True)
    except:
        print("Phi-2 failed to load (likely VRAM). Falling back to GPT-2-Large.")
        eval_model_name = 'gpt2-large'
        phi2_model = AutoModelForCausalLM.from_pretrained(eval_model_name).to(device)
        phi2_tokenizer = AutoTokenizer.from_pretrained(eval_model_name)

    phi2_model.eval()

    print(f"\nEvaluating {len(samples)} generated samples...")

    perplexities = []
    
    with torch.no_grad():
        for i, text in enumerate(samples):
            tokens = phi2_tokenizer.encode(text, return_tensors='pt').to(device)

            if tokens.size(1) < 2:
                continue

            outputs = phi2_model(tokens, labels=tokens)
            loss = outputs.loss
            ppl = torch.exp(loss).item()
            
            # Filter out crazy outliers for the plot
            if ppl < 1000:
                perplexities.append(ppl)

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(samples)} samples...")

    avg_ppl = np.mean(perplexities) if perplexities else 0.0

    print(f"\n=== Generation Quality ({eval_model_name} Perplexity) ===")
    print(f"  Average Perplexity: {avg_ppl:.2f}")

    if plot_histogram and perplexities:
        plt.figure(figsize=(10, 6))
        plt.hist(perplexities, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        plt.axvline(avg_ppl, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {avg_ppl:.1f}')
        plt.title('Distribution of Generation Quality')
        plt.xlabel('Perplexity (Lower is Better)')
        plt.ylabel('Count of Samples')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('generation_quality_hist.png')
        print("Saved histogram to 'generation_quality_hist.png'")

    return avg_ppl, perplexities