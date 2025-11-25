"""
Training script for Vision Transformer emotion recognition model.
Supports training, validation, and model checkpointing.
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.vit_emotion_model import ViTEmotionClassifier
from data.dataloader import EmotionDataset, DirectoryDataset, FER2013Dataset, create_data_loaders
from data.transforms import get_train_transforms, get_val_transforms


class EmotionTrainer:
    """Trainer class for emotion recognition model."""
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 optimizer: optim.Optimizer,
                 criterion: nn.Module,
                 device: torch.device,
                 save_dir: str = './outputs',
                 log_dir: str = './outputs/logs',
                 emotion_labels: Optional[List[str]] = None):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer for training
            criterion: Loss function
            device: Device to train on
            save_dir: Directory to save checkpoints
            log_dir: Directory for tensorboard logs
            emotion_labels: List of emotion label names
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.emotion_labels = emotion_labels or ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        # Create directories
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize tensorboard writer
        self.writer = SummaryWriter(log_dir)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_val_acc = 0.0
        self.best_epoch = 0
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_file = os.path.join(self.save_dir, 'training.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Log progress
            if batch_idx % 100 == 0:
                self.logger.info(f'Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}, '
                               f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, epoch: int) -> Tuple[float, float, np.ndarray, List[int], List[int]]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        return avg_loss, accuracy, cm, all_labels, all_predictions
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'emotion_labels': self.emotion_labels
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            self.logger.info(f'New best model saved with validation accuracy: {self.best_val_acc:.2f}%')
    
    def plot_confusion_matrix(self, cm: np.ndarray, epoch: int):
        """Plot and save confusion matrix."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.emotion_labels,
                   yticklabels=self.emotion_labels)
        plt.title(f'Confusion Matrix - Epoch {epoch}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plot_path = os.path.join(self.save_dir, f'confusion_matrix_epoch_{epoch}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def train(self, num_epochs: int, save_every: int = 5, validate_every: int = 1):
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
            validate_every: Validate every N epochs
        """
        self.logger.info(f'Starting training for {num_epochs} epochs')
        self.logger.info(f'Device: {self.device}')
        self.logger.info(f'Model: {self.model.__class__.__name__}')
        self.logger.info(f'Training samples: {len(self.train_loader.dataset)}')
        self.logger.info(f'Validation samples: {len(self.val_loader.dataset)}')
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            # Training
            train_loss, train_acc = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validation
            if epoch % validate_every == 0:
                val_loss, val_acc, cm, val_labels, val_predictions = self.validate(epoch)
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_acc)
                
                # Check if best model
                is_best = val_acc > self.best_val_acc
                if is_best:
                    self.best_val_acc = val_acc
                    self.best_epoch = epoch
                
                # Log metrics
                epoch_time = time.time() - epoch_start
                self.logger.info(f'Epoch {epoch}/{num_epochs} - Time: {epoch_time:.2f}s')
                self.logger.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
                self.logger.info(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
                
                # Tensorboard logging
                self.writer.add_scalar('Loss/Train', train_loss, epoch)
                self.writer.add_scalar('Loss/Val', val_loss, epoch)
                self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
                self.writer.add_scalar('Accuracy/Val', val_acc, epoch)
                
                # Save confusion matrix
                self.plot_confusion_matrix(cm, epoch)
                
                # Classification report
                report = classification_report(val_labels, val_predictions, 
                                             target_names=self.emotion_labels, 
                                             output_dict=True)
                
                # Log per-class metrics
                for emotion, metrics in report.items():
                    if isinstance(metrics, dict):
                        self.writer.add_scalar(f'F1/{emotion}', metrics.get('f1-score', 0), epoch)
                        self.writer.add_scalar(f'Precision/{emotion}', metrics.get('precision', 0), epoch)
                        self.writer.add_scalar(f'Recall/{emotion}', metrics.get('recall', 0), epoch)
                
                # Save checkpoint
                if epoch % save_every == 0 or is_best:
                    self.save_checkpoint(epoch, is_best)
        
        # Training completed
        total_time = time.time() - start_time
        self.logger.info(f'Training completed in {total_time:.2f}s')
        self.logger.info(f'Best validation accuracy: {self.best_val_acc:.2f}% (Epoch {self.best_epoch})')
        
        # Save final model
        self.save_checkpoint(num_epochs, False)
        
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch
        }
        
        history_path = os.path.join(self.save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        self.writer.close()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train ViT Emotion Recognition Model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--dataset_type', type=str, choices=['directory', 'fer2013'], default='directory',
                       help='Type of dataset format')
    parser.add_argument('--model_name', type=str, default='google/vit-base-patch16-224',
                       help='Pretrained ViT model name')
    parser.add_argument('--num_classes', type=int, default=7, help='Number of emotion classes')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--freeze_backbone', action='store_true', help='Freeze ViT backbone')
    parser.add_argument('--save_dir', type=str, default='./outputs/checkpoints', help='Directory to save models')
    parser.add_argument('--log_dir', type=str, default='./outputs/logs', help='Directory for tensorboard logs')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
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
    
    # Data transforms
    train_transform = get_train_transforms(use_augmentation=True)
    val_transform = get_val_transforms()
    
    # Load datasets
    if args.dataset_type == 'directory':
        full_dataset = DirectoryDataset.from_directory(args.data_dir, transform=None)
        emotion_labels = full_dataset.emotion_labels
        
        # Split dataset
        from data.dataloader import split_dataset
        train_dataset, val_dataset, test_dataset = split_dataset(full_dataset)
        
        # Apply transforms
        train_dataset.transform = train_transform
        val_dataset.transform = val_transform
        
    elif args.dataset_type == 'fer2013':
        train_dataset = FER2013Dataset.from_csv(
            os.path.join(args.data_dir, 'fer2013.csv'),
            os.path.join(args.data_dir, 'images'),
            transform=train_transform,
            split='Training'
        )
        val_dataset = FER2013Dataset.from_csv(
            os.path.join(args.data_dir, 'fer2013.csv'),
            os.path.join(args.data_dir, 'images'),
            transform=val_transform,
            split='PublicTest'
        )
        emotion_labels = train_dataset.emotion_labels
    
    # Create data loaders
    loaders = create_data_loaders(
        train_dataset, val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_balanced_sampling=True
    )
    
    # Model
    model = ViTEmotionClassifier(
        model_name=args.model_name,
        num_classes=args.num_classes,
        dropout_rate=args.dropout_rate,
        freeze_backbone=args.freeze_backbone
    ).to(device)
    
    print(f'Model info: {model.get_model_info()}')
    
    # Optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Trainer
    trainer = EmotionTrainer(
        model=model,
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        emotion_labels=emotion_labels
    )
    
    # Train
    trainer.train(num_epochs=args.num_epochs)


if __name__ == '__main__':
    main()
