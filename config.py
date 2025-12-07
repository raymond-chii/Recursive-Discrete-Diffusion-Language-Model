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
    'num_timesteps': 100
}

# Training hyperparameters
TRAINING_CONFIG = {
    'batch_size': 64,  # Reduced for more complex data
    'num_epochs': 10,
    'learning_rate': 5e-5,
    'recursion_depth': 3,
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
    'recursion_depth': 10,
    'temperature': 0.9
}
