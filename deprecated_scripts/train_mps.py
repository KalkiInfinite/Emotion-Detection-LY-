"""
Optimized training script for Apple Silicon with MPS acceleration.
This version includes MPS-specific optimizations for faster training.
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
from data.dataloader import EmotionDataset, DirectoryDataset, create_data_loaders
from data.transforms import get_train_transforms, get_val_transforms


def get_optimal_device():
    """Get the optimal device with MPS priority for Apple Silicon."""
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f'🚀 Using MPS (Apple Silicon GPU) for acceleration')
        return device, True  # is_mps = True
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'🚀 Using CUDA GPU for acceleration')
        return device, False
    else:
        device = torch.device('cpu')
        print(f'⚠️ Using CPU (training will be slow)')
        return device, False


def optimize_for_mps(model, device, is_mps=False):
    """Apply MPS-specific optimizations."""
    model = model.to(device)
    
    if is_mps:
        print("🔧 Applying MPS optimizations...")
        # MPS works best with specific data types
        # Note: Mixed precision is limited on MPS, so we'll use float32
        print("   - Using float32 precision for MPS stability")
        
        # Ensure model is in the right precision
        model = model.float()
    
    return model


def create_optimized_data_loaders(train_dataset, val_dataset, batch_size=32, num_workers=4, is_mps=False):
    """Create optimized data loaders for MPS."""
    
    # MPS works better with specific configurations
    if is_mps:
        # Reduce num_workers for MPS to avoid overhead
        num_workers = min(num_workers, 2)
        # Don't use pin_memory with MPS
        pin_memory = False
        print(f"🔧 MPS optimization: num_workers={num_workers}, pin_memory={pin_memory}")
    else:
        pin_memory = True
    
    loaders = create_data_loaders(
        train_dataset, val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        use_balanced_sampling=True
    )
    
    # Update pin_memory setting
    for loader_name, loader in loaders.items():
        loader.pin_memory = pin_memory
    
    return loaders


class OptimizedEmotionTrainer:
    """Optimized trainer for Apple Silicon MPS."""
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 optimizer: optim.Optimizer,
                 criterion: nn.Module,
                 device: torch.device,
                 is_mps: bool = False,
                 save_dir: str = './outputs',
                 log_dir: str = './outputs/logs',
                 emotion_labels: Optional[List[str]] = None):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.is_mps = is_mps
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.emotion_labels = emotion_labels or ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
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
        
        # MPS specific optimizations
        if self.is_mps:
            print("🔧 Initialized MPS-optimized trainer")
    
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
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train for one epoch with MPS optimizations."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Progress tracking
        batch_count = len(self.train_loader)
        log_interval = max(1, batch_count // 10)  # Log 10 times per epoch
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            # Move to device
            images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
            
            # Ensure correct dtype for MPS
            if self.is_mps:
                images = images.float()
                labels = labels.long()
            
            # Forward pass
            self.optimizer.zero_grad()
            
            try:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Log progress
                if batch_idx % log_interval == 0 or batch_idx == batch_count - 1:
                    progress = (batch_idx + 1) / batch_count * 100
                    self.logger.info(f'Epoch {epoch}, Batch {batch_idx+1}/{batch_count} ({progress:.1f}%), '
                                   f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
                
            except RuntimeError as e:
                if "out of memory" in str(e) or "MPS" in str(e):
                    self.logger.warning(f"Memory issue at batch {batch_idx}, skipping...")
                    if self.is_mps:
                        torch.mps.empty_cache()
                    continue
                else:
                    raise e
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, epoch: int) -> Tuple[float, float, np.ndarray, List[int], List[int]]:
        """Validate the model with MPS optimizations."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.val_loader):
                images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                
                # Ensure correct dtype for MPS
                if self.is_mps:
                    images = images.float()
                    labels = labels.long()
                
                try:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    
                    total_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
                except RuntimeError as e:
                    if "out of memory" in str(e) or "MPS" in str(e):
                        self.logger.warning(f"Memory issue in validation batch {batch_idx}, skipping...")
                        if self.is_mps:
                            torch.mps.empty_cache()
                        continue
                    else:
                        raise e
        
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
            'emotion_labels': self.emotion_labels,
            'device_used': str(self.device)
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            self.logger.info(f'🏆 New best model saved with validation accuracy: {self.best_val_acc:.2f}%')
    
    def train(self, num_epochs: int, save_every: int = 5):
        """Main training loop with MPS optimizations."""
        self.logger.info(f'🚀 Starting MPS-accelerated training for {num_epochs} epochs')
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
            
            # Clear MPS cache after training epoch
            if self.is_mps:
                torch.mps.empty_cache()
            
            # Validation
            val_loss, val_acc, cm, val_labels, val_predictions = self.validate(epoch)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Clear MPS cache after validation
            if self.is_mps:
                torch.mps.empty_cache()
            
            # Check if best model
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
            
            # Log metrics
            epoch_time = time.time() - epoch_start
            self.logger.info(f'✅ Epoch {epoch}/{num_epochs} - Time: {epoch_time:.2f}s')
            self.logger.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            self.logger.info(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Tensorboard logging
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/Val', val_acc, epoch)
            
            # Save checkpoint
            if epoch % save_every == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
        
        # Training completed
        total_time = time.time() - start_time
        self.logger.info(f'🎉 Training completed in {total_time:.2f}s')
        self.logger.info(f'🏆 Best validation accuracy: {self.best_val_acc:.2f}% (Epoch {self.best_epoch})')
        
        # Save final model and history
        self.save_checkpoint(num_epochs, False)
        
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'total_time': total_time,
            'device_used': str(self.device)
        }
        
        history_path = os.path.join(self.save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        self.writer.close()


def main():
    """Main training function optimized for MPS."""
    parser = argparse.ArgumentParser(description='MPS-Optimized ViT Emotion Recognition Training')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--dataset_type', type=str, choices=['directory', 'pre_split'], default='directory')
    parser.add_argument('--model_name', type=str, default='google/vit-base-patch16-224')
    parser.add_argument('--num_classes', type=int, default=8, help='Number of emotion classes')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (will be optimized for MPS)')
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--save_dir', type=str, default='./outputs/mps_training')
    parser.add_argument('--log_dir', type=str, default='./outputs/mps_logs')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--force_cpu', action='store_true', help='Force CPU usage')
    
    args = parser.parse_args()
    
    # Get optimal device
    if args.force_cpu:
        device = torch.device('cpu')
        is_mps = False
        print('🐌 Forced CPU usage')
    else:
        device, is_mps = get_optimal_device()
    
    # Optimize batch size for MPS
    if is_mps and args.batch_size > 64:
        args.batch_size = 64
        print(f"🔧 Reduced batch size to {args.batch_size} for MPS optimization")
    
    # Data transforms
    train_transform = get_train_transforms(use_augmentation=True)
    val_transform = get_val_transforms()
    
    # Load datasets based on type
    if args.dataset_type == 'pre_split':
        # Load pre-split datasets
        train_dataset = DirectoryDataset.from_directory(
            os.path.join(args.data_dir, 'train'), 
            transform=train_transform
        )
        val_dataset = DirectoryDataset.from_directory(
            os.path.join(args.data_dir, 'val'), 
            transform=val_transform
        )
        emotion_labels = train_dataset.emotion_labels
    else:
        # Regular directory dataset with splitting
        full_dataset = DirectoryDataset.from_directory(args.data_dir, transform=None)
        emotion_labels = full_dataset.emotion_labels
        
        from data.dataloader import split_dataset
        train_dataset, val_dataset, _ = split_dataset(full_dataset)
        train_dataset.transform = train_transform
        val_dataset.transform = val_transform
    
    print(f"📊 Dataset loaded: {len(train_dataset)} train, {len(val_dataset)} val")
    print(f"🎭 Emotions: {emotion_labels}")
    
    # Create optimized data loaders
    loaders = create_optimized_data_loaders(
        train_dataset, val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        is_mps=is_mps
    )
    
    # Model
    model = ViTEmotionClassifier(
        model_name=args.model_name,
        num_classes=len(emotion_labels),
        dropout_rate=0.1
    )
    
    # Optimize model for device
    model = optimize_for_mps(model, device, is_mps)
    
    print(f'📊 Model info: {model.get_model_info()}')
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Trainer
    trainer = OptimizedEmotionTrainer(
        model=model,
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        is_mps=is_mps,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        emotion_labels=emotion_labels
    )
    
    # Train
    trainer.train(num_epochs=args.num_epochs)


if __name__ == '__main__':
    main()
