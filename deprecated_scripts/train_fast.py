"""
Fixed and optimized MPS training with proper progress reporting.
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
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.vit_emotion_model import ViTEmotionClassifier
from data.dataloader import EmotionDataset, DirectoryDataset, create_data_loaders
from data.transforms import get_train_transforms, get_val_transforms


class FastMPSTrainer:
    """Optimized trainer with frequent progress updates."""
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 optimizer: optim.Optimizer,
                 criterion: nn.Module,
                 device: torch.device,
                 save_dir: str = './outputs',
                 emotion_labels: Optional[List[str]] = None):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.save_dir = save_dir
        self.emotion_labels = emotion_labels or ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
        # Create directories
        os.makedirs(save_dir, exist_ok=True)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_val_acc = 0.0
        self.best_epoch = 0
        
        print(f"🚀 FastMPSTrainer initialized")
        print(f"   Device: {device}")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train for one epoch with frequent progress updates."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        batch_count = len(self.train_loader)
        # Log every 50 batches OR every 5% of batches, whichever is smaller
        log_interval = min(50, max(1, batch_count // 20))
        
        print(f"\n📊 Epoch {epoch} - Training on {batch_count} batches (logging every {log_interval} batches)")
        print("-" * 80)
        
        epoch_start_time = time.time()
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            batch_start_time = time.time()
            
            # Move to device (non-blocking for speed)
            images = images.to(self.device, non_blocking=True).float()
            labels = labels.to(self.device, non_blocking=True).long()
            
            # Forward pass
            self.optimizer.zero_grad()
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
            
            batch_time = time.time() - batch_start_time
            
            # Frequent progress logging
            if batch_idx % log_interval == 0 or batch_idx == batch_count - 1:
                progress = (batch_idx + 1) / batch_count * 100
                avg_loss = total_loss / (batch_idx + 1)
                accuracy = 100. * correct / total
                
                # Estimate time remaining
                elapsed = time.time() - epoch_start_time
                if batch_idx > 0:
                    estimated_total = elapsed * batch_count / (batch_idx + 1)
                    eta = estimated_total - elapsed
                    eta_str = f"ETA: {eta/60:.1f}min"
                else:
                    eta_str = "ETA: calculating..."
                
                print(f"Batch {batch_idx+1:4d}/{batch_count} ({progress:5.1f}%) | "
                      f"Loss: {loss.item():.4f} | Acc: {accuracy:5.1f}% | "
                      f"Speed: {batch_time:.2f}s/batch | {eta_str}")
                
                # Clear MPS cache periodically
                if batch_idx % 100 == 0 and str(self.device) == 'mps':
                    torch.mps.empty_cache()
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        epoch_time = time.time() - epoch_start_time
        
        print("-" * 80)
        print(f"✅ Epoch {epoch} Training Complete: Loss={avg_loss:.4f}, Acc={accuracy:.2f}%, Time={epoch_time/60:.1f}min")
        
        return avg_loss, accuracy
    
    def validate(self, epoch: int) -> Tuple[float, float]:
        """Validate the model quickly."""
        print(f"\n🔍 Validating epoch {epoch}...")
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        val_start_time = time.time()
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.val_loader):
                images = images.to(self.device, non_blocking=True).float()
                labels = labels.to(self.device, non_blocking=True).long()
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Show validation progress every 20 batches
                if batch_idx % 20 == 0:
                    progress = (batch_idx + 1) / len(self.val_loader) * 100
                    print(f"   Val batch {batch_idx+1:3d}/{len(self.val_loader)} ({progress:4.1f}%)")
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        val_time = time.time() - val_start_time
        
        print(f"✅ Validation Complete: Loss={avg_loss:.4f}, Acc={accuracy:.2f}%, Time={val_time:.1f}s")
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch: int, val_acc: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'best_val_acc': self.best_val_acc,
            'emotion_labels': self.emotion_labels,
            'device_used': str(self.device)
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"🏆 NEW BEST MODEL! Validation accuracy: {val_acc:.2f}%")
    
    def train(self, num_epochs: int):
        """Main training loop with excellent progress reporting."""
        print(f"\n🚀 STARTING FAST MPS TRAINING")
        print("=" * 80)
        print(f"Epochs: {num_epochs}")
        print(f"Training samples: {len(self.train_loader.dataset):,}")
        print(f"Validation samples: {len(self.val_loader.dataset):,}")
        print(f"Device: {self.device}")
        print("=" * 80)
        
        training_start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n🔥 EPOCH {epoch}/{num_epochs}")
            print("=" * 50)
            
            # Training
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Clear cache after training
            if str(self.device) == 'mps':
                torch.mps.empty_cache()
            
            # Validation
            val_loss, val_acc = self.validate(epoch)
            
            # Clear cache after validation
            if str(self.device) == 'mps':
                torch.mps.empty_cache()
            
            # Track best model
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_acc, is_best)
            
            # Summary
            print(f"\n📊 EPOCH {epoch} SUMMARY:")
            print(f"   Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
            print(f"   Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
            print(f"   Best:  {self.best_val_acc:.2f}% (Epoch {self.best_epoch})")
            
            # Time estimation
            elapsed = time.time() - training_start_time
            if epoch > 1:
                avg_time_per_epoch = elapsed / epoch
                eta_total = avg_time_per_epoch * (num_epochs - epoch)
                print(f"   ETA:   {eta_total/60:.1f} minutes remaining")
        
        # Training completed
        total_time = time.time() - training_start_time
        print(f"\n🎉 TRAINING COMPLETED!")
        print("=" * 80)
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}% (Epoch {self.best_epoch})")
        print(f"Final model saved to: {self.save_dir}")


def main():
    """Main function with optimized settings."""
    parser = argparse.ArgumentParser(description='Fast MPS Training')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--dataset_type', type=str, choices=['directory', 'pre_split'], default='pre_split')
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)  # Smaller for speed
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--save_dir', type=str, default='./outputs/fast_mps')
    parser.add_argument('--use_cpu', action='store_true', help='Force CPU usage')
    
    args = parser.parse_args()
    
    # Device selection
    if args.use_cpu:
        device = torch.device('cpu')
        print("🐌 Using CPU (forced)")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("🚀 Using MPS (Apple Silicon GPU)")
    else:
        device = torch.device('cpu')
        print("🐌 Using CPU (MPS not available)")
    
    # Load datasets
    print(f"\n📁 Loading datasets from {args.data_dir}...")
    
    train_transform = get_train_transforms(use_augmentation=True)
    val_transform = get_val_transforms()
    
    if args.dataset_type == 'pre_split':
        train_dataset = DirectoryDataset.from_directory(
            os.path.join(args.data_dir, 'train'), 
            transform=train_transform
        )
        val_dataset = DirectoryDataset.from_directory(
            os.path.join(args.data_dir, 'val'), 
            transform=val_transform
        )
    else:
        full_dataset = DirectoryDataset.from_directory(args.data_dir, transform=None)
        from data.dataloader import split_dataset
        train_dataset, val_dataset, _ = split_dataset(full_dataset)
        train_dataset.transform = train_transform
        val_dataset.transform = val_transform
    
    emotion_labels = train_dataset.emotion_labels
    print(f"✅ Loaded {len(train_dataset):,} train, {len(val_dataset):,} val images")
    print(f"🎭 Emotions: {emotion_labels}")
    
    # Optimized data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=1,  # Single worker for MPS
        pin_memory=False  # Don't use pin_memory with MPS
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=1,
        pin_memory=False
    )
    
    # Model
    print(f"\n🤖 Creating model...")
    model = ViTEmotionClassifier(
        model_name='google/vit-base-patch16-224',
        num_classes=len(emotion_labels),
        dropout_rate=0.1
    ).to(device).float()
    
    print(f"✅ Model loaded: {model.get_model_info()}")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Trainer
    trainer = FastMPSTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        save_dir=args.save_dir,
        emotion_labels=emotion_labels
    )
    
    # Start training
    trainer.train(num_epochs=args.num_epochs)


if __name__ == '__main__':
    main()
