import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm.auto import tqdm
from utils import sample


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
    ttr = unique_tokens / total_tokens

    # Bigrams
    bigrams = [' '.join(all_tokens[i:i+2]) for i in range(len(all_tokens)-1)]
    unique_bigrams = len(set(bigrams))

    print(f"\nDiversity Metrics:")
    print(f"  Unique tokens: {unique_tokens} / {total_tokens}")
    print(f"  Type-Token Ratio: {ttr:.3f}")
    print(f"  Unique bigrams: {unique_bigrams}")

    # Most common words
    counter = Counter(all_tokens)
    print(f"\nMost common words:")
    for word, count in counter.most_common(10):
        print(f"  '{word}': {count} times")

    return samples


def compute_perplexity(model, diffusion, tokenizer, num_test_samples=1000, device='cuda', dataset_name='openwebtext'):
    """
    Compute diffusion model perplexity on reconstruction task.
    Uses FIXED timestep for consistent evaluation.
    """
    print("Loading test data...")
    # OpenWebText doesn't have validation split, use train with offset
    if dataset_name == 'openwebtext':
        test_dataset = load_dataset(dataset_name, split='train', streaming=True)
        # Skip first samples to get "validation-like" data
        test_dataset = test_dataset.skip(8000000)
    else:
        test_dataset = load_dataset(dataset_name, split='validation', streaming=True)

    test_samples = []
    for i, example in enumerate(test_dataset):
        if i >= num_test_samples:
            break
        test_samples.append(example['text'])

    print(f"Loaded {len(test_samples)} test samples")

    # Compute perplexity at fixed timestep
    model.eval()
    total_loss = 0
    total_tokens = 0
    fixed_timestep = 500  # Middle of diffusion process

    with torch.no_grad():
        for text in test_samples:
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
            logits = model(x_t, t, recursion_depth=2)

            # Loss
            loss = F.cross_entropy(
                logits[mask].reshape(-1, logits.size(-1)),
                tokens[mask].reshape(-1),
                reduction='sum'
            )
            total_loss += loss.item()
            total_tokens += mask.sum().item()

    perplexity = np.exp(total_loss / total_tokens)

    print(f"\n=== Diffusion Model Perplexity (t={fixed_timestep}) ===")
    print(f"  Perplexity: {perplexity:.2f}")
    print(f"  (Lower is better)")
    print(f"  Random baseline: ~{len(tokenizer)}")

    return perplexity


def evaluate_generation_quality_phi2(samples, device='cuda'):
    """
    Evaluate generated text quality using Microsoft Phi-2 (2.7B params).
    Phi-2 is a state-of-the-art small model, much better than GPT-2.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading Phi-2 (2.7B) for quality evaluation...")
    print("(First run will download ~5GB model)")

    phi2_model = AutoModelForCausalLM.from_pretrained(
        'microsoft/phi-2',
        torch_dtype=torch.float16,  # Use FP16 to save memory
        trust_remote_code=True
    ).to(device)
    phi2_tokenizer = AutoTokenizer.from_pretrained('microsoft/phi-2', trust_remote_code=True)
    phi2_model.eval()

    print(f"\nEvaluating {len(samples[:20])} generated samples with Phi-2...")

    total_logprob = 0
    total_tokens = 0

    with torch.no_grad():
        for i, text in enumerate(samples[:20]):
            tokens = phi2_tokenizer.encode(text, return_tensors='pt').to(device)

            if tokens.size(1) < 2:
                continue

            outputs = phi2_model(tokens, labels=tokens)
            loss = outputs.loss

            total_logprob += -loss.item() * tokens.size(1)
            total_tokens += tokens.size(1)

            if (i + 1) % 5 == 0:
                print(f"  Processed {i + 1}/20 samples...")

    phi2_perplexity = np.exp(-total_logprob / total_tokens)

    print(f"\n=== Generation Quality (Phi-2 Evaluation) ===")
    print(f"  Phi-2 Perplexity: {phi2_perplexity:.2f}")
    print(f"  (Lower = more human-like/coherent)")
    print(f"  Interpretation:")
    print(f"    <15  = Excellent, human-like quality")
    print(f"    15-30  = Good quality")
    print(f"    30-60  = Acceptable")
    print(f"    >60  = Poor/incoherent")

    return phi2_perplexity


def test_diverse_prompts(model, diffusion, tokenizer, max_length=128, recursion_depth=10,
                        temperature=0.9, device='cuda'):
    diverse_prompts = [
        "The future of artificial intelligence",
        "In recent years, scientists have discovered",
        "According to the latest research",
        "Many people believe that",
        "The internet has changed",
        "Technology companies are",
        "When it comes to politics",
        "Climate change is",
        "The economy",
        "In a surprising turn of events"
    ]

    print("Generating samples from diverse prompts...\n")

    for prompt in diverse_prompts:
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
        print(f"{prompt}: {text}")
        print()
