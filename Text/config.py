"""
Configuration settings for the emotion detection project.
"""

import os

# Model configurations
MODEL_CONFIGS = {
    'bert-base': {
        'model_name': 'bert-base-uncased',
        'max_length': 128,
        'batch_size': 16,
        'learning_rate': 2e-5,
        'epochs': 3
    },
    'roberta-base': {
        'model_name': 'roberta-base',
        'max_length': 128,
        'batch_size': 16,
        'learning_rate': 2e-5,
        'epochs': 3
    },
    'distilbert': {
        'model_name': 'distilbert-base-uncased',
        'max_length': 128,
        'batch_size': 32,
        'learning_rate': 3e-5,
        'epochs': 4
    }
}

# Dataset configuration
DATASET_CONFIG = {
    'name': 'dair-ai/emotion',
    'emotion_labels': {
        0: "sadness",
        1: "joy", 
        2: "love",
        3: "anger",
        4: "fear",
        5: "surprise"
    }
}

# Training configuration
TRAINING_CONFIG = {
    'output_dir': './emotion_model_results',
    'save_dir': './saved_emotion_model',
    'eval_strategy': 'epoch',
    'save_strategy': 'epoch',
    'warmup_steps': 500,
    'weight_decay': 0.01,
    'logging_steps': 100,
    'save_total_limit': 2,
    'seed': 42,
    'metric_for_best_model': 'eval_f1',
    'greater_is_better': True,
    'load_best_model_at_end': True
}

# Device configuration
def get_device():
    """Get the best available device (GPU/CPU)."""
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
PLOTS_DIR = os.path.join(PROJECT_ROOT, 'plots')

# Create directories if they don't exist
for directory in [RESULTS_DIR, MODELS_DIR, PLOTS_DIR]:
    os.makedirs(directory, exist_ok=True)
