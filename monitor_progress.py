"""
Training Monitor - Check training progress in real-time
"""

import json
import time
import os
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def load_training_history(checkpoint_dir="./outputs/safe_training"):
    """Load training history."""
    history_path = Path(checkpoint_dir) / "training_history.json"
    
    if not history_path.exists():
        return None
    
    try:
        with open(history_path, 'r') as f:
            return json.load(f)
    except:
        return None

def print_current_status(checkpoint_dir="./outputs/safe_training"):
    """Print current training status."""
    checkpoint_path = Path(checkpoint_dir) / "latest_checkpoint.pt"
    
    if not checkpoint_path.exists():
        print("❌ No active training found")
        return
    
    # Load checkpoint
    import torch
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        history = load_training_history(checkpoint_dir)
        
        print("📊 CURRENT TRAINING STATUS")
        print("=" * 50)
        print(f"Epoch: {checkpoint['epoch'] + 1} / {checkpoint.get('max_epochs', '?')}")
        print(f"Batch: {checkpoint['batch'] + 1}")
        print(f"Total batches trained: {checkpoint.get('total_batches_trained', 0):,}")
        print(f"Best validation accuracy: {checkpoint.get('best_val_acc', 0):.2f}%")
        print(f"Current learning rate: {checkpoint.get('current_lr', 'unknown')}")
        print(f"Device: {checkpoint.get('device_used', 'unknown')}")
        print(f"Last updated: {checkpoint.get('timestamp', 'unknown')}")
        
        if history and 'training_history' in history:
            recent_entries = history['training_history'][-5:]  # Last 5 entries
            print(f"\n📈 RECENT TRAINING HISTORY (last {len(recent_entries)} entries):")
            print("-" * 50)
            for entry in recent_entries:
                print(f"Epoch {entry['epoch']}, Batch {entry['batch']}: "
                      f"Loss={entry['loss']:.4f}, Acc={entry.get('accuracy', 0):.2f}%")
        
    except Exception as e:
        print(f"❌ Error reading checkpoint: {e}")

def plot_training_progress(checkpoint_dir="./outputs/safe_training", save_plot=True):
    """Plot training progress."""
    history = load_training_history(checkpoint_dir)
    
    if not history or 'training_history' not in history:
        print("❌ No training history found")
        return
    
    data = history['training_history']
    if len(data) < 2:
        print("❌ Not enough data to plot")
        return
    
    try:
        import matplotlib.pyplot as plt
        
        # Extract data
        batches = [entry['batch'] for entry in data]
        losses = [entry['loss'] for entry in data]
        accuracies = [entry.get('accuracy', 0) for entry in data]
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot loss
        ax1.plot(batches, losses, 'b-', linewidth=1, alpha=0.7)
        ax1.set_title('Training Loss Over Time')
        ax1.set_xlabel('Batch')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        
        # Plot accuracy
        ax2.plot(batches, accuracies, 'g-', linewidth=1, alpha=0.7)
        ax2.set_title('Training Accuracy Over Time')
        ax2.set_xlabel('Batch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = Path(checkpoint_dir) / "training_progress.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"📊 Training plot saved to: {plot_path}")
        
        plt.show()
        
    except ImportError:
        print("❌ Matplotlib not available for plotting")
    except Exception as e:
        print(f"❌ Error creating plot: {e}")

def monitor_continuous(checkpoint_dir="./outputs/safe_training", interval=30):
    """Monitor training progress continuously."""
    print("📡 CONTINUOUS TRAINING MONITOR")
    print("=" * 50)
    print(f"Checking every {interval} seconds...")
    print("Press Ctrl+C to stop monitoring")
    print()
    
    try:
        while True:
            os.system('clear' if os.name == 'posix' else 'cls')  # Clear screen
            print(f"🕒 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print_current_status(checkpoint_dir)
            
            print(f"\n⏰ Next update in {interval} seconds...")
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n👋 Stopped monitoring")

def main():
    """Main function."""
    print("📊 TRAINING PROGRESS MONITOR")
    print("=" * 40)
    
    print("Options:")
    print("1. Check current status")
    print("2. Plot training progress")
    print("3. Monitor continuously (every 30s)")
    print("4. Monitor continuously (every 10s)")
    
    try:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            print_current_status()
        elif choice == '2':
            plot_training_progress()
        elif choice == '3':
            monitor_continuous(interval=30)
        elif choice == '4':
            monitor_continuous(interval=10)
        else:
            print("❌ Invalid choice")
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == '__main__':
    main()
