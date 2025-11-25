"""
Evaluation script for trained ViT emotion recognition model.
Provides detailed metrics, confusion matrix, and per-class analysis.
"""

import os
import sys
import argparse
import json
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.vit_emotion_model import ViTEmotionClassifier
from data.dataloader import EmotionDataset, DirectoryDataset, FER2013Dataset, create_data_loaders
from data.transforms import get_val_transforms


class EmotionEvaluator:
    """Evaluator class for emotion recognition model."""
    
    def __init__(self, 
                 model: nn.Module,
                 test_loader: DataLoader,
                 device: torch.device,
                 emotion_labels: List[str],
                 save_dir: str = './outputs/evaluation'):
        """
        Initialize evaluator.
        
        Args:
            model: Trained PyTorch model
            test_loader: Test data loader
            device: Device to run evaluation on
            emotion_labels: List of emotion label names
            save_dir: Directory to save evaluation results
        """
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.emotion_labels = emotion_labels
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Results storage
        self.predictions = []
        self.true_labels = []
        self.probabilities = []
    
    def evaluate(self) -> Dict:
        """
        Evaluate the model on test data.
        
        Returns:
            Dictionary containing evaluation metrics
        """
        self.model.eval()
        correct = 0
        total = 0
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        print("Running evaluation...")
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.test_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                # Collect results
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                # Statistics
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                if batch_idx % 50 == 0:
                    print(f"Processed {batch_idx}/{len(self.test_loader)} batches")
        
        # Store results
        self.predictions = np.array(all_predictions)
        self.true_labels = np.array(all_labels)
        self.probabilities = np.array(all_probabilities)
        
        # Calculate metrics
        metrics = self.calculate_metrics()
        
        print(f"Evaluation completed. Overall accuracy: {metrics['accuracy']:.4f}")
        
        return metrics
    
    def calculate_metrics(self) -> Dict:
        """Calculate comprehensive evaluation metrics."""
        # Basic metrics
        accuracy = accuracy_score(self.true_labels, self.predictions)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            self.true_labels, self.predictions, average=None, zero_division=0
        )
        
        # Macro and weighted averages
        precision_macro = precision_recall_fscore_support(
            self.true_labels, self.predictions, average='macro', zero_division=0
        )[0]
        recall_macro = precision_recall_fscore_support(
            self.true_labels, self.predictions, average='macro', zero_division=0
        )[1]
        f1_macro = precision_recall_fscore_support(
            self.true_labels, self.predictions, average='macro', zero_division=0
        )[2]
        
        precision_weighted = precision_recall_fscore_support(
            self.true_labels, self.predictions, average='weighted', zero_division=0
        )[0]
        recall_weighted = precision_recall_fscore_support(
            self.true_labels, self.predictions, average='weighted', zero_division=0
        )[1]
        f1_weighted = precision_recall_fscore_support(
            self.true_labels, self.predictions, average='weighted', zero_division=0
        )[2]
        
        # Confusion matrix
        cm = confusion_matrix(self.true_labels, self.predictions)
        
        # Per-class metrics dictionary
        per_class_metrics = {}
        for i, emotion in enumerate(self.emotion_labels):
            per_class_metrics[emotion] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i]),
                'support': int(support[i])
            }
        
        metrics = {
            'accuracy': float(accuracy),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'f1_macro': float(f1_macro),
            'precision_weighted': float(precision_weighted),
            'recall_weighted': float(recall_weighted),
            'f1_weighted': float(f1_weighted),
            'per_class_metrics': per_class_metrics,
            'confusion_matrix': cm.tolist(),
            'total_samples': len(self.true_labels)
        }
        
        return metrics
    
    def plot_confusion_matrix(self, normalize: bool = False, save_path: str = None):
        """
        Plot confusion matrix.
        
        Args:
            normalize: Whether to normalize the confusion matrix
            save_path: Path to save the plot
        """
        cm = confusion_matrix(self.true_labels, self.predictions)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = 'Normalized Confusion Matrix'
            fmt = '.2f'
        else:
            title = 'Confusion Matrix'
            fmt = 'd'
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=self.emotion_labels,
                   yticklabels=self.emotion_labels)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path is None:
            save_path = os.path.join(self.save_dir, f'confusion_matrix{"_normalized" if normalize else ""}.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved to {save_path}")
    
    def plot_per_class_metrics(self, save_path: str = None):
        """Plot per-class precision, recall, and F1-score."""
        metrics_data = []
        emotions = []
        
        for emotion in self.emotion_labels:
            if emotion in [m for m in self.calculate_metrics()['per_class_metrics']]:
                metrics = self.calculate_metrics()['per_class_metrics'][emotion]
                metrics_data.append([
                    metrics['precision'],
                    metrics['recall'],
                    metrics['f1_score']
                ])
                emotions.append(emotion)
        
        metrics_data = np.array(metrics_data)
        
        # Create plot
        x = np.arange(len(emotions))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars1 = ax.bar(x - width, metrics_data[:, 0], width, label='Precision', alpha=0.8)
        bars2 = ax.bar(x, metrics_data[:, 1], width, label='Recall', alpha=0.8)
        bars3 = ax.bar(x + width, metrics_data[:, 2], width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Emotions')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Performance Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(emotions, rotation=45)
        ax.legend()
        ax.set_ylim(0, 1.1)
        
        # Add value labels on bars
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=8)
        
        add_labels(bars1)
        add_labels(bars2)
        add_labels(bars3)
        
        if save_path is None:
            save_path = os.path.join(self.save_dir, 'per_class_metrics.png')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Per-class metrics plot saved to {save_path}")
    
    def plot_class_distribution(self, save_path: str = None):
        """Plot class distribution in test set."""
        unique, counts = np.unique(self.true_labels, return_counts=True)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar([self.emotion_labels[i] for i in unique], counts, alpha=0.7)
        plt.title('Class Distribution in Test Set')
        plt.xlabel('Emotions')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom')
        
        if save_path is None:
            save_path = os.path.join(self.save_dir, 'class_distribution.png')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Class distribution plot saved to {save_path}")
    
    def save_results(self, metrics: Dict, save_path: str = None):
        """Save evaluation results to JSON file."""
        if save_path is None:
            save_path = os.path.join(self.save_dir, 'evaluation_results.json')
        
        # Add classification report
        report = classification_report(
            self.true_labels, self.predictions,
            target_names=self.emotion_labels,
            output_dict=True
        )
        
        results = {
            'metrics': metrics,
            'classification_report': report,
            'emotion_labels': self.emotion_labels
        }
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Evaluation results saved to {save_path}")
    
    def print_summary(self, metrics: Dict):
        """Print evaluation summary."""
        print("\n" + "="*60)
        print("EMOTION RECOGNITION MODEL EVALUATION SUMMARY")
        print("="*60)
        
        print(f"Total test samples: {metrics['total_samples']}")
        print(f"Overall accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro F1-score: {metrics['f1_macro']:.4f}")
        print(f"Weighted F1-score: {metrics['f1_weighted']:.4f}")
        
        print(f"\nMacro averages:")
        print(f"  Precision: {metrics['precision_macro']:.4f}")
        print(f"  Recall: {metrics['recall_macro']:.4f}")
        print(f"  F1-score: {metrics['f1_macro']:.4f}")
        
        print(f"\nWeighted averages:")
        print(f"  Precision: {metrics['precision_weighted']:.4f}")
        print(f"  Recall: {metrics['recall_weighted']:.4f}")
        print(f"  F1-score: {metrics['f1_weighted']:.4f}")
        
        print(f"\nPer-class performance:")
        for emotion, metrics_dict in metrics['per_class_metrics'].items():
            print(f"  {emotion:>10}: P={metrics_dict['precision']:.3f}, "
                  f"R={metrics_dict['recall']:.3f}, F1={metrics_dict['f1_score']:.3f}, "
                  f"Support={metrics_dict['support']}")
        
        print("="*60)


def load_model(checkpoint_path: str, device: torch.device) -> Tuple[nn.Module, List[str]]:
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model info from checkpoint
    emotion_labels = checkpoint.get('emotion_labels', 
                                  ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'])
    num_classes = len(emotion_labels)
    
    # Create model
    model = ViTEmotionClassifier(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"Best validation accuracy: {checkpoint.get('best_val_acc', 'N/A')}")
    print(f"Training epoch: {checkpoint.get('epoch', 'N/A')}")
    
    return model, emotion_labels


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate ViT Emotion Recognition Model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to test dataset directory')
    parser.add_argument('--dataset_type', type=str, choices=['directory', 'fer2013'], default='directory',
                       help='Type of dataset format')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--save_dir', type=str, default='./outputs/evaluation', help='Directory to save results')
    parser.add_argument('--split', type=str, default='test', help='Dataset split to evaluate (for FER2013)')
    
    args = parser.parse_args()
    
    # Device - prioritize MPS for Apple Silicon Macs
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f'Using device: MPS (Apple Silicon GPU)')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'Using device: CUDA GPU')
    else:
        device = torch.device('cpu')
        print(f'Using device: CPU')
    
    print(f'Device: {device}')
    
    # Load model
    model, emotion_labels = load_model(args.checkpoint, device)
    
    # Data transforms
    test_transform = get_val_transforms()
    
    # Load test dataset
    if args.dataset_type == 'directory':
        test_dataset = DirectoryDataset.from_directory(args.data_dir, transform=test_transform)
    elif args.dataset_type == 'fer2013':
        test_dataset = FER2013Dataset.from_csv(
            os.path.join(args.data_dir, 'fer2013.csv'),
            os.path.join(args.data_dir, 'images'),
            transform=test_transform,
            split=args.split
        )
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Evaluator
    evaluator = EmotionEvaluator(
        model=model,
        test_loader=test_loader,
        device=device,
        emotion_labels=emotion_labels,
        save_dir=args.save_dir
    )
    
    # Run evaluation
    metrics = evaluator.evaluate()
    
    # Generate plots and save results
    evaluator.plot_confusion_matrix(normalize=False)
    evaluator.plot_confusion_matrix(normalize=True)
    evaluator.plot_per_class_metrics()
    evaluator.plot_class_distribution()
    evaluator.save_results(metrics)
    evaluator.print_summary(metrics)


if __name__ == '__main__':
    main()
