from transformers import GPT2Tokenizer
from config import MODEL_CONFIG

def count_params(config):
    d_model = config['d_model']
    n_heads = config['n_heads']
    n_layers = config['n_layers']
    max_seq_len = config['max_seq_len']

    # Vocab size (GPT2 tokenizer + 1 mask token)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    vocab_size = len(tokenizer) + 1

    print("=" * 60)
    print("TINY RECURSIVE TRANSFORMER - PARAMETER COUNT")
    print("=" * 60)
    print(f"\nConfig:")
    print(f"  vocab_size: {vocab_size:,}")
    print(f"  d_model: {d_model}")
    print(f"  n_heads: {n_heads}")
    print(f"  n_layers: {n_layers}")
    print(f"  max_seq_len: {max_seq_len}")

    print(f"\n{'Component':<30} {'Parameters':>15}")
    print("-" * 60)

    total = 0

    # 1. Token Embeddings
    token_embed = vocab_size * d_model
    print(f"{'Token Embeddings':<30} {token_embed:>15,}")
    total += token_embed

    # 2. Position Embeddings
    pos_embed = max_seq_len * d_model
    print(f"{'Position Embeddings':<30} {pos_embed:>15,}")
    total += pos_embed

    # 3. Timestep Embedding (3 linear layers)
    time_embed = (d_model * d_model) + d_model + (d_model * d_model) + d_model
    print(f"{'Timestep Embedding':<30} {time_embed:>15,}")
    total += time_embed

    # 4. Input Projection
    input_proj = (d_model * d_model) + d_model
    print(f"{'Input Projection':<30} {input_proj:>15,}")
    total += input_proj

    # 5. Transformer Blocks (reused n_layers times in recursion)
    mha_params = 4 * (d_model * d_model + d_model)  # Q, K, V, O projections

    # 2 LayerNorms (gamma + beta for each)
    ln_params = 2 * (d_model + d_model)

    # MLP: d_model -> 4*d_model -> d_model
    mlp_params = (d_model * 4 * d_model) + (4 * d_model) + (4 * d_model * d_model) + d_model

    block_params = mha_params + ln_params + mlp_params
    total_blocks_params = block_params * n_layers

    print(f"{'  - MultiheadAttention (per block)':<30} {mha_params:>15,}")
    print(f"{'  - LayerNorms (per block)':<30} {ln_params:>15,}")
    print(f"{'  - MLP (per block)':<30} {mlp_params:>15,}")
    print(f"{'Transformer Blocks (x{n_layers})':<30} {total_blocks_params:>15,}")
    total += total_blocks_params

    # 6. Final LayerNorm
    ln_f = d_model + d_model
    print(f"{'Final LayerNorm':<30} {ln_f:>15,}")
    total += ln_f

    # 7. Output Head
    output_head = (d_model * vocab_size) + vocab_size
    print(f"{'Output Head':<30} {output_head:>15,}")
    total += output_head

    print("-" * 60)
    print(f"{'TOTAL PARAMETERS':<30} {total:>15,}")
    print("=" * 60)

    # Show effective compute with recursion
    print(f"\nWith recursion_depth=6:")
    print(f"  Effective transformer passes: {n_layers * 6}")
    print(f"  (But parameters are REUSED, not duplicated!)")

    # Compare to non-recursive model
    equivalent_layers = n_layers * 6
    equivalent_params = total - total_blocks_params + (block_params * equivalent_layers)
    print(f"\nEquivalent non-recursive model:")
    print(f"  Would need {equivalent_layers} layers")
    print(f"  Would have {equivalent_params:,} parameters")
    print(f"  Parameter efficiency: {equivalent_params / total:.2f}x")

    return total

if __name__ == "__main__":
    total_params = count_params(MODEL_CONFIG)

    # Show in millions
    print(f"\n✨ Model size: {total_params / 1_000_000:.2f}M parameters")
