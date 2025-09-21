"""
Utility functions for data preprocessing, visualization, and model evaluation.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import torch
from sklearn.metrics import classification_report, confusion_matrix


def create_directories(dirs):
    """Create directories if they don't exist."""
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created directory: {directory}")


def save_training_history(history, save_path):
    """Save training history to JSON file."""
    try:
        with open(save_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"💾 Training history saved to: {save_path}")
    except Exception as e:
        print(f"❌ Error saving training history: {str(e)}")


def plot_training_metrics(trainer, save_path="training_metrics.png"):
    """Plot training and validation metrics."""
    try:
        log_history = trainer.state.log_history
        
        train_losses = []
        eval_losses = []
        eval_accuracies = []
        eval_f1_scores = []
        epochs = []
        
        for log in log_history:
            if 'train_loss' in log:
                train_losses.append(log['train_loss'])
            if 'eval_loss' in log:
                eval_losses.append(log['eval_loss'])
                eval_accuracies.append(log['eval_accuracy'])
                eval_f1_scores.append(log['eval_f1'])
                epochs.append(log['epoch'])
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training loss
        ax1.plot(train_losses, 'b-', label='Training Loss')
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Validation loss
        ax2.plot(epochs, eval_losses, 'r-', label='Validation Loss')
        ax2.set_title('Validation Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        # Validation accuracy
        ax3.plot(epochs, eval_accuracies, 'g-', label='Validation Accuracy')
        ax3.set_title('Validation Accuracy')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy')
        ax3.legend()
        ax3.grid(True)
        
        # Validation F1 score
        ax4.plot(epochs, eval_f1_scores, 'm-', label='Validation F1 Score')
        ax4.set_title('Validation F1 Score')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('F1 Score')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Training metrics plot saved to: {save_path}")
        plt.show()
        
    except Exception as e:
        print(f"⚠️ Could not create training metrics plot: {str(e)}")


def plot_emotion_distribution(dataset, emotion_labels, save_path="emotion_distribution.png"):
    """Plot the distribution of emotions in the dataset."""
    try:
        labels = [sample['label'] for sample in dataset]
        
        # Count occurrences
        label_counts = {}
        for label in labels:
            emotion_name = emotion_labels[label]
            label_counts[emotion_name] = label_counts.get(emotion_name, 0) + 1
        
        # Create bar plot
        plt.figure(figsize=(12, 6))
        emotions = list(label_counts.keys())
        counts = list(label_counts.values())
        
        bars = plt.bar(emotions, counts, color=plt.cm.Set3(np.linspace(0, 1, len(emotions))))
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom')
        
        plt.title('Distribution of Emotions in Dataset')
        plt.xlabel('Emotion')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Emotion distribution plot saved to: {save_path}")
        plt.show()
        
    except Exception as e:
        print(f"⚠️ Could not create emotion distribution plot: {str(e)}")


def create_detailed_confusion_matrix(y_true, y_pred, emotion_labels, save_path="detailed_confusion_matrix.png"):
    """Create a detailed confusion matrix with percentages."""
    try:
        cm = confusion_matrix(y_true, y_pred)
        cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=list(emotion_labels.values()),
                   yticklabels=list(emotion_labels.values()), ax=ax1)
        ax1.set_title('Confusion Matrix (Raw Counts)')
        ax1.set_xlabel('Predicted Emotion')
        ax1.set_ylabel('True Emotion')
        
        # Percentages
        sns.heatmap(cm_percentage, annot=True, fmt='.1f', cmap='Reds',
                   xticklabels=list(emotion_labels.values()),
                   yticklabels=list(emotion_labels.values()), ax=ax2)
        ax2.set_title('Confusion Matrix (Percentages)')
        ax2.set_xlabel('Predicted Emotion')
        ax2.set_ylabel('True Emotion')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Detailed confusion matrix saved to: {save_path}")
        plt.show()
        
    except Exception as e:
        print(f"⚠️ Could not create detailed confusion matrix: {str(e)}")


def print_model_info(model, tokenizer):
    """Print detailed information about the model and tokenizer."""
    try:
        print("\n🤖 MODEL INFORMATION")
        print("=" * 50)
        
        # Model info
        print(f"Model Name: {model.config.name_or_path}")
        print(f"Model Type: {model.config.model_type}")
        print(f"Number of Parameters: {model.num_parameters():,}")
        print(f"Number of Labels: {model.config.num_labels}")
        print(f"Hidden Size: {model.config.hidden_size}")
        
        if hasattr(model.config, 'num_attention_heads'):
            print(f"Attention Heads: {model.config.num_attention_heads}")
        if hasattr(model.config, 'num_hidden_layers'):
            print(f"Hidden Layers: {model.config.num_hidden_layers}")
        
        # Tokenizer info
        print(f"\n📝 TOKENIZER INFORMATION")
        print("=" * 50)
        print(f"Tokenizer Type: {type(tokenizer).__name__}")
        print(f"Vocabulary Size: {tokenizer.vocab_size}")
        print(f"Model Max Length: {tokenizer.model_max_length}")
        
        # Special tokens
        special_tokens = {
            'pad_token': tokenizer.pad_token,
            'unk_token': tokenizer.unk_token,
            'cls_token': getattr(tokenizer, 'cls_token', None),
            'sep_token': getattr(tokenizer, 'sep_token', None),
            'mask_token': getattr(tokenizer, 'mask_token', None)
        }
        
        print("\nSpecial Tokens:")
        for token_name, token_value in special_tokens.items():
            if token_value:
                print(f"  {token_name}: {token_value}")
        
    except Exception as e:
        print(f"⚠️ Could not display model info: {str(e)}")


def get_device_info():
    """Get detailed information about the computing device."""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"\n💻 DEVICE INFORMATION")
        print("=" * 50)
        print(f"Device: {device}")
        
        if torch.cuda.is_available():
            print(f"GPU Count: {torch.cuda.device_count()}")
            print(f"Current GPU: {torch.cuda.current_device()}")
            print(f"GPU Name: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print(f"CUDA Version: {torch.version.cuda}")
        else:
            print("CUDA not available. Using CPU.")
            
        return device
        
    except Exception as e:
        print(f"⚠️ Could not get device info: {str(e)}")
        return torch.device("cpu")


def create_prediction_report(predictions, save_path="prediction_report.json"):
    """Create and save a detailed prediction report."""
    try:
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_predictions': len(predictions),
            'predictions': predictions
        }
        
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📄 Prediction report saved to: {save_path}")
        return report
        
    except Exception as e:
        print(f"❌ Error creating prediction report: {str(e)}")


def validate_text_input(text, max_length=512):
    """Validate and clean text input."""
    if not isinstance(text, str):
        raise ValueError("Input must be a string")
    
    if len(text.strip()) == 0:
        raise ValueError("Input text cannot be empty")
    
    if len(text) > max_length:
        print(f"⚠️ Text truncated from {len(text)} to {max_length} characters")
        text = text[:max_length]
    
    return text.strip()


def format_prediction_output(prediction_result):
    """Format prediction results for better display."""
    try:
        text = prediction_result['text']
        predicted_emotion = prediction_result['predicted_emotion']
        confidence = prediction_result['confidence']
        top_predictions = prediction_result['top_predictions']
        
        output = f"""
{'='*60}
📝 Text: "{text[:100]}{'...' if len(text) > 100 else ''}"
🎭 Predicted Emotion: {predicted_emotion.upper()}
🎯 Confidence: {confidence:.3f} ({confidence*100:.1f}%)

🏆 Top 3 Predictions:
"""
        for i, (emotion, prob) in enumerate(top_predictions, 1):
            bar_length = int(prob * 20)  # Scale to 20 characters
            bar = '█' * bar_length + '░' * (20 - bar_length)
            output += f"   {i}. {emotion:<8} {bar} {prob:.3f} ({prob*100:.1f}%)\n"
        
        return output
        
    except Exception as e:
        return f"❌ Error formatting prediction: {str(e)}"
