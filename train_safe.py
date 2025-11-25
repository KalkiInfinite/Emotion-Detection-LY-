"""
Ultra-Safe Training with Batch-Level Checkpointing
Saves after every batch and allows graceful interruption with user input.
"""

import os
import sys
import argparse
import json
import time
import signal
import threading
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

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.vit_emotion_model import ViTEmotionClassifier
from data.dataloader import EmotionDataset, DirectoryDataset, create_data_loaders
from data.transforms import get_train_transforms, get_val_transforms


class SafeTrainer:
    """Ultra-safe trainer with batch-level checkpointing."""
    
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
        self.emotion_labels = emotion_labels or ['angry', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
        # Training state
        self.current_epoch = 0
        self.current_batch = 0
        self.total_batches_trained = 0
        self.best_val_acc = 0.0
        self.training_history = []
        self.should_stop = False
        
        # Create directories
        os.makedirs(save_dir, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Setup graceful interruption
        self.setup_interruption_handling()
        
        # Checkpoint file paths
        self.checkpoint_path = os.path.join(save_dir, 'latest_checkpoint.pt')
        self.best_model_path = os.path.join(save_dir, 'best_model.pt')
        self.history_path = os.path.join(save_dir, 'training_history.json')
        
        self.logger.info("✅ SafeTrainer initialized with batch-level checkpointing")
    
    def print_device_info(self):
        """Print detailed device information and usage."""
        device_type = str(self.device).upper()
        
        if self.device.type == 'mps':
            print(f"   🚀 Using Apple Silicon MPS (Metal Performance Shaders)")
            try:
                # Check MPS availability and status
                print(f"   📊 MPS Built: {torch.backends.mps.is_built()}")
                print(f"   📊 MPS Available: {torch.backends.mps.is_available()}")
                
                # Test MPS with a small operation
                test_tensor = torch.randn(2, 2).to(self.device)
                _ = test_tensor @ test_tensor
                print(f"   ✅ MPS GPU operations: Working")
                
            except Exception as e:
                print(f"   ⚠️ MPS GPU test failed: {e}")
                
        elif self.device.type == 'cuda':
            print(f"   🚀 Using NVIDIA CUDA GPU")
            try:
                gpu_name = torch.cuda.get_device_name(self.device)
                memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(self.device) / 1024**3
                print(f"   📊 GPU: {gpu_name}")
                print(f"   📊 Memory Allocated: {memory_allocated:.2f} GB")
                print(f"   📊 Memory Reserved: {memory_reserved:.2f} GB")
            except Exception as e:
                print(f"   ⚠️ CUDA info unavailable: {e}")
                
        else:
            print(f"   💻 Using CPU")
            try:
                import psutil
                cpu_count = psutil.cpu_count(logical=True)
                memory_gb = psutil.virtual_memory().total / 1024**3
                print(f"   📊 CPU Cores: {cpu_count}")
                print(f"   📊 System Memory: {memory_gb:.1f} GB")
            except ImportError:
                print(f"   📊 CPU info: Install psutil for detailed stats")
    
    def log_device_usage(self, batch_idx: int):
        """Log device usage during training."""
        if batch_idx % 200 == 0:  # Log every 200 batches
            if self.device.type == 'mps':
                try:
                    # Clear cache and report
                    torch.mps.empty_cache()
                    print(f"   🔄 MPS cache cleared (batch {batch_idx})")
                except:
                    pass
            elif self.device.type == 'cuda':
                try:
                    memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3
                    memory_reserved = torch.cuda.memory_reserved(self.device) / 1024**3
                    print(f"   📊 GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
                    if memory_allocated > 8:  # If using > 8GB, clear cache
                        torch.cuda.empty_cache()
                        print(f"   🔄 CUDA cache cleared")
                except:
                    pass
            else:
                try:
                    import psutil
                    cpu_percent = psutil.cpu_percent(interval=0.1)
                    memory_percent = psutil.virtual_memory().percent
                    print(f"   📊 CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}%")
                except ImportError:
                    pass
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_file = os.path.join(self.save_dir, 'safe_training.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_interruption_handling(self):
        """Setup graceful interruption handling."""
        # Handle Ctrl+C
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # Start input monitoring thread
        self.input_thread = threading.Thread(target=self.monitor_input, daemon=True)
        self.input_thread.start()
    
    def signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        print("\n🛑 Interrupt signal received! Saving checkpoint...")
        self.should_stop = True
    
    def monitor_input(self):
        """Monitor for user input to stop training."""
        while not self.should_stop:
            try:
                user_input = input().strip().lower()
                if user_input in ['stop', 'quit', 'exit', 's', 'q']:
                    print("\n🛑 User requested stop! Saving checkpoint...")
                    self.should_stop = True
                    break
            except:
                break
    
    def save_checkpoint(self, batch_idx: int, epoch: int, is_best: bool = False, force_save: bool = False):
        """Save checkpoint after each batch."""
        checkpoint = {
            'epoch': epoch,
            'batch': batch_idx,
            'total_batches_trained': self.total_batches_trained,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'training_history': self.training_history,
            'emotion_labels': self.emotion_labels,
            'device_used': str(self.device),
            'timestamp': str(datetime.now())
        }
        
        # Save latest checkpoint (every batch or when forced)
        if batch_idx % 10 == 0 or force_save or self.should_stop:  # Save every 10 batches
            torch.save(checkpoint, self.checkpoint_path)
            if batch_idx % 100 == 0 or force_save:
                self.logger.info(f"💾 Checkpoint saved at epoch {epoch}, batch {batch_idx}")
        
        # Save best model
        if is_best:
            torch.save(checkpoint, self.best_model_path)
            self.logger.info(f"🏆 New best model saved! Val accuracy: {self.best_val_acc:.2f}%")
        
        # Save history
        with open(self.history_path, 'w') as f:
            json.dump({
                'training_history': self.training_history,
                'current_epoch': epoch,
                'current_batch': batch_idx,
                'total_batches_trained': self.total_batches_trained,
                'best_val_acc': self.best_val_acc,
                'last_updated': str(datetime.now())
            }, f, indent=2)
    
    def load_checkpoint(self):
        """Load checkpoint if exists."""
        if os.path.exists(self.checkpoint_path):
            try:
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
                
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.current_epoch = checkpoint['epoch']
                self.current_batch = checkpoint['batch']
                self.total_batches_trained = checkpoint.get('total_batches_trained', 0)
                self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
                self.training_history = checkpoint.get('training_history', [])
                
                self.logger.info(f"📂 Resumed from epoch {self.current_epoch}, batch {self.current_batch}")
                self.logger.info(f"📊 Total batches trained so far: {self.total_batches_trained}")
                self.logger.info(f"🏆 Best validation accuracy: {self.best_val_acc:.2f}%")
                
                return True
            except Exception as e:
                self.logger.warning(f"⚠️ Could not load checkpoint: {e}")
                return False
        return False
    
    def validate_quick(self) -> float:
        """Quick validation on a subset for frequent checking."""
        self.model.eval()
        correct = 0
        total = 0
        samples_to_check = min(1000, len(self.val_loader.dataset))  # Max 1000 samples
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.val_loader):
                if total >= samples_to_check:
                    break
                    
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100. * correct / total if total > 0 else 0.0
        return accuracy
    
    def train_batch(self, images, labels):
        """Train on a single batch."""
        self.model.train()
        
        # Move to device
        images, labels = images.to(self.device), labels.to(self.device)
        
        # Forward pass
        self.optimizer.zero_grad()
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        accuracy = 100. * correct / labels.size(0)
        
        return loss.item(), accuracy
    
    def train(self, max_epochs: int = 50, validate_every: int = 500):
        """Main training loop with batch-level checkpointing."""
        
        # Try to resume from checkpoint
        resumed = self.load_checkpoint()
        if resumed:
            print(f"🔄 Resuming training from epoch {self.current_epoch}, batch {self.current_batch}")
        else:
            print(f"🚀 Starting fresh training")
        
        print(f"\n📋 Training Configuration:")
        print(f"   Device: {self.device}")
        self.print_device_info()
        print(f"   Training samples: {len(self.train_loader.dataset):,}")
        print(f"   Validation samples: {len(self.val_loader.dataset):,}")
        print(f"   Batch size: {self.train_loader.batch_size}")
        print(f"   Batches per epoch: {len(self.train_loader):,}")
        print(f"   Target epochs: {max_epochs}")
        
        print(f"\n🎮 Controls:")
        print(f"   Type 'stop', 'quit', 'exit', 's', or 'q' and press Enter to stop safely")
        print(f"   Or press Ctrl+C for immediate stop with checkpoint save")
        print(f"   Progress saves automatically every 10 batches")
        
        start_time = time.time()
        total_batches = len(self.train_loader)
        
        # Start from where we left off
        start_epoch = self.current_epoch
        if self.current_batch >= total_batches:
            start_epoch += 1
            self.current_batch = 0
        
        try:
            for epoch in range(start_epoch, max_epochs):
                self.current_epoch = epoch
                epoch_start_time = time.time()
                
                print(f"\n{'='*60}")
                print(f"🎯 EPOCH {epoch + 1}/{max_epochs}")
                print(f"{'='*60}")
                
                # Skip batches if resuming mid-epoch
                start_batch = self.current_batch if epoch == start_epoch else 0
                
                for batch_idx, (images, labels) in enumerate(self.train_loader):
                    if batch_idx < start_batch:
                        continue  # Skip already processed batches
                    
                    if self.should_stop:
                        print(f"\n🛑 Training stopped by user at epoch {epoch}, batch {batch_idx}")
                        break
                    
                    # Train on batch
                    loss, acc = self.train_batch(images, labels)
                    self.current_batch = batch_idx
                    self.total_batches_trained += 1
                    
                    # Progress reporting
                    if batch_idx % 50 == 0 or batch_idx == total_batches - 1:
                        elapsed = time.time() - start_time
                        progress = (batch_idx + 1) / total_batches * 100
                        eta_epoch = (time.time() - epoch_start_time) / (batch_idx + 1 - start_batch) * (total_batches - batch_idx - 1)
                        
                        print(f"📊 Epoch {epoch+1} | Batch {batch_idx+1:4}/{total_batches} ({progress:5.1f}%) | "
                              f"Loss: {loss:.4f} | Acc: {acc:5.1f}% | "
                              f"ETA: {eta_epoch/60:.1f}min | "
                              f"Total batches: {self.total_batches_trained:,}")
                    
                    # Log device usage periodically
                    self.log_device_usage(batch_idx)
                    
                    # Save checkpoint
                    self.save_checkpoint(batch_idx, epoch)
                    
                    # Quick validation check
                    if batch_idx % validate_every == 0 and batch_idx > 0:
                        val_acc = self.validate_quick()
                        is_best = val_acc > self.best_val_acc
                        if is_best:
                            self.best_val_acc = val_acc
                        
                        print(f"🔍 Quick validation: {val_acc:.2f}% {'🏆 NEW BEST!' if is_best else ''}")
                        self.save_checkpoint(batch_idx, epoch, is_best=is_best)
                    
                    # Note: Device-specific cache clearing is now handled in log_device_usage()
                
                if self.should_stop:
                    break
                
                # End of epoch
                self.current_batch = 0  # Reset for next epoch
                epoch_time = time.time() - epoch_start_time
                
                # Full validation at end of epoch
                val_acc = self.validate_quick()
                is_best = val_acc > self.best_val_acc
                if is_best:
                    self.best_val_acc = val_acc
                
                print(f"\n✅ Epoch {epoch+1} completed in {epoch_time/60:.1f} minutes")
                print(f"🎯 Validation accuracy: {val_acc:.2f}% {'🏆 NEW BEST!' if is_best else ''}")
                
                # Save end-of-epoch checkpoint
                self.save_checkpoint(total_batches - 1, epoch, is_best=is_best, force_save=True)
                
                # Update history
                self.training_history.append({
                    'epoch': epoch + 1,
                    'val_accuracy': val_acc,
                    'epoch_time': epoch_time,
                    'total_batches_trained': self.total_batches_trained
                })
        
        except Exception as e:
            self.logger.error(f"❌ Training error: {e}")
            print(f"💾 Saving emergency checkpoint...")
            self.save_checkpoint(self.current_batch, self.current_epoch, force_save=True)
            raise
        
        finally:
            # Final save
            total_time = time.time() - start_time
            print(f"\n🎉 Training session completed!")
            print(f"⏰ Total time: {total_time/3600:.1f} hours")
            print(f"📊 Total batches trained: {self.total_batches_trained:,}")
            print(f"🏆 Best validation accuracy: {self.best_val_acc:.2f}%")
            
            self.save_checkpoint(self.current_batch, self.current_epoch, force_save=True)
            print(f"💾 Final checkpoint saved to {self.checkpoint_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Ultra-Safe ViT Training with Batch Checkpointing')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--dataset_type', type=str, choices=['directory', 'pre_split'], default='pre_split')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--save_dir', type=str, default='./outputs/safe_training')
    parser.add_argument('--validate_every', type=int, default=500, help='Validate every N batches')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f'🚀 Using MPS (Apple Silicon GPU)')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'🚀 Using CUDA GPU')
    else:
        device = torch.device('cpu')
        print(f'⚠️ Using CPU')
    
    # Data loading
    train_transform = get_train_transforms(use_augmentation=True)
    val_transform = get_val_transforms()
    
    if args.dataset_type == 'pre_split':
        train_dataset = DirectoryDataset.from_directory(
            os.path.join(args.data_dir, 'train'), transform=train_transform
        )
        val_dataset = DirectoryDataset.from_directory(
            os.path.join(args.data_dir, 'val'), transform=val_transform
        )
    else:
        full_dataset = DirectoryDataset.from_directory(args.data_dir, transform=None)
        from data.dataloader import split_dataset
        train_dataset, val_dataset, _ = split_dataset(full_dataset)
        train_dataset.transform = train_transform
        val_dataset.transform = val_transform
    
    emotion_labels = train_dataset.emotion_labels
    
    # Data loaders
    loaders = create_data_loaders(
        train_dataset, val_dataset,
        batch_size=args.batch_size,
        num_workers=2,  # Keep low for MPS
        use_balanced_sampling=True
    )
    
    # Model
    model = ViTEmotionClassifier(
        model_name='google/vit-base-patch16-224',
        num_classes=len(emotion_labels),
        dropout_rate=0.1
    ).to(device)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Trainer
    trainer = SafeTrainer(
        model=model,
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        save_dir=args.save_dir,
        emotion_labels=emotion_labels
    )
    
    # Start training
    trainer.train(max_epochs=args.max_epochs, validate_every=args.validate_every)


if __name__ == '__main__':
    main()
