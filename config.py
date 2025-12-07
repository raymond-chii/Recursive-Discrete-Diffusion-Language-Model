import torch

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model hyperparameters
MODEL_CONFIG = {
    'd_model': 516,
    'n_heads': 6,
    'n_layers': 6,
    'max_seq_len': 128,
    'dropout': 0.1
}

# Diffusion hyperparameters
DIFFUSION_CONFIG = {
    'num_timesteps': 50  # Reduced from 100 to minimize error accumulation during sampling
}

# Training hyperparameters
TRAINING_CONFIG = {
    'batch_size': 64,  # Reduced for more complex data
    'num_epochs': 10,
    'learning_rate': 1e-4,  # Increased from 5e-5 for faster learning
    'recursion_depth': 6,  # Increased from 3 for better iterative refinement
    'gradient_clip': 1.0
}

# Dataset hyperparameters
DATASET_CONFIG = {
    'dataset_name': 'Skylion007/openwebtext',  # Updated OpenWebText location
    'max_length': 128,  # Increased for web text
    'num_samples': 500000  # Increased for better coverage
}

# Sampling hyperparameters
SAMPLING_CONFIG = {
    'max_length': 128,  # Match training length
    'recursion_depth': 8,  # Reduced from 10, more aligned with training depth
    'temperature': 1.0  # Slightly increased for more diversity
}
