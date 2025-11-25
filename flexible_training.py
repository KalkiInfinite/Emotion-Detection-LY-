"""
Ultra-Flexible Training Controller - Perfect for your workflow!
Start with 1 epoch, stop anytime, test results, resume with any target!
"""

import os
import sys
import json
from pathlib import Path

def check_checkpoint_status(checkpoint_dir="./outputs/safe_training"):
    """Check the current training status."""
    checkpoint_path = Path(checkpoint_dir) / "latest_checkpoint.pt"
    history_path = Path(checkpoint_dir) / "training_history.json"
    
    print("🔍 TRAINING STATUS CHECK")
    print("=" * 50)
    
    if not checkpoint_path.exists():
        print("❌ No checkpoint found. Starting fresh.")
        return False, {}
    
    # Load checkpoint info
    import torch
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print(f"📂 Checkpoint found!")
        print(f"   Current epoch: {checkpoint['epoch'] + 1}")
        print(f"   Current batch: {checkpoint['batch'] + 1}")
        print(f"   Total batches trained so far: {checkpoint.get('total_batches_trained', 0):,}")
        print(f"   Best validation accuracy: {checkpoint.get('best_val_acc', 0):.2f}%")
        
        # Calculate progress
        current_epoch = checkpoint['epoch'] + 1
        current_batch = checkpoint['batch'] + 1
        
        # Load history if available
        if history_path.exists():
            with open(history_path, 'r') as f:
                history = json.load(f)
                print(f"   Training history entries: {len(history.get('training_history', []))}")
                
                # Show recent progress
                if 'training_history' in history and history['training_history']:
                    recent = history['training_history'][-3:]
                    print(f"   Recent progress:")
                    for entry in recent:
                        if 'val_accuracy' in entry:
                            print(f"     Epoch {entry['epoch']}: {entry['val_accuracy']:.2f}% accuracy")
        
        return True, {
            'current_epoch': current_epoch,
            'current_batch': current_batch,
            'best_val_acc': checkpoint.get('best_val_acc', 0),
            'total_batches_trained': checkpoint.get('total_batches_trained', 0)
        }
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return False, {}

def start_training(target_epochs, resume=False):
    """Start training with specified target epochs."""
    
    action = "Resuming" if resume else "Starting"
    print(f"\n🚀 {action.upper()} TRAINING")
    print("=" * 60)
    print(f"🎯 Target epochs: {target_epochs}")
    
    # Calculate estimates
    batches_per_epoch = 3126  # From your dataset (100,022 / 32)
    estimated_time_per_epoch_low = 0.4  # hours
    estimated_time_per_epoch_high = 0.9  # hours
    estimated_time_low = target_epochs * estimated_time_per_epoch_low
    estimated_time_high = target_epochs * estimated_time_per_epoch_high
    
    print(f"⏱️  Estimated time: {estimated_time_low:.1f}-{estimated_time_high:.1f} hours")
    print(f"📊 Batches per epoch: {batches_per_epoch:,}")
    print(f"💾 Auto-save every 50 batches (~2-3 minutes)")
    print(f"🔍 Validation every 300 batches (~10-15 minutes)")
    
    # Build command
    cmd = f"""
cd /Users/piyushtyagi/FacialEmotion && \\
/Users/piyushtyagi/FacialEmotion/.venv/bin/python train_safe.py \\
    --data_dir data/unified_dataset \\
    --dataset_type pre_split \\
    --batch_size 32 \\
    --max_epochs {target_epochs} \\
    --learning_rate 1e-4 \\
    --validate_every 300"""
    
    if resume:
        cmd += " \\\n    --resume"
    
    print(f"\n🎮 TRAINING CONTROLS:")
    print(f"   • Type 'stop', 'quit', 'exit', 's', or 'q' and press Enter to stop safely")
    print(f"   • Progress automatically saved every 50 batches")
    print(f"   • You can resume later with any new epoch target")
    print(f"   • Test your model anytime using saved checkpoints")
    
    print(f"\n🔧 Command being executed:")
    print(cmd.strip())
    
    confirm = input(f"\n✅ Start training? (y/N): ").strip().lower()
    if confirm == 'y':
        print(f"\n🚀 Starting training...")
        print(f"💡 Remember: Type 'stop' and press Enter to exit safely!")
        os.system(cmd)
    else:
        print("❌ Training cancelled")

def test_model():
    """Test the current model."""
    print(f"\n🧪 TESTING CURRENT MODEL")
    print("=" * 50)
    
    # Check if checkpoint exists
    checkpoint_path = Path("./outputs/safe_training/latest_checkpoint.pt")
    if not checkpoint_path.exists():
        print("❌ No model checkpoint found. Train first!")
        return
    
    cmd = """
cd /Users/piyushtyagi/FacialEmotion && \\
/Users/piyushtyagi/FacialEmotion/.venv/bin/python eval.py \\
    --model_path outputs/safe_training/latest_checkpoint.pt \\
    --data_dir data/unified_dataset \\
    --dataset_type pre_split
"""
    
    print("🔍 Running evaluation on test set...")
    print("📊 This will show accuracy, confusion matrix, and per-class metrics")
    print(f"\nCommand being executed:")
    print(cmd.strip())
    
    confirm = input(f"\n✅ Run evaluation? (y/N): ").strip().lower()
    if confirm == 'y':
        os.system(cmd)
    else:
        print("❌ Evaluation cancelled")

def quick_inference():
    """Run quick inference on sample images."""
    print(f"\n🎯 QUICK INFERENCE TEST")
    print("=" * 50)
    
    cmd = """
cd /Users/piyushtyagi/FacialEmotion && \\
/Users/piyushtyagi/FacialEmotion/.venv/bin/python inference.py \\
    --model_path outputs/safe_training/latest_checkpoint.pt \\
    --data_dir data/unified_dataset/test \\
    --num_samples 10
"""
    
    print("🔍 Testing on 10 random images from test set...")
    print(f"\nCommand being executed:")
    print(cmd.strip())
    
    confirm = input(f"\n✅ Run quick test? (y/N): ").strip().lower()
    if confirm == 'y':
        os.system(cmd)
    else:
        print("❌ Quick test cancelled")

def main():
    """Main function."""
    print("🎛️  ULTRA-FLEXIBLE TRAINING CONTROL")
    print("=" * 60)
    print("💡 PERFECT WORKFLOW:")
    print("   1️⃣  Start with 1 epoch to test everything works")
    print("   2️⃣  Stop and test the model performance") 
    print("   3️⃣  Resume with more epochs based on your available time")
    print("   4️⃣  Repeat: stop anytime, test, continue as needed")
    print("   🔄 Never lose progress - resumes from exact batch!")
    
    # Check if checkpoint exists
    has_checkpoint, checkpoint_info = check_checkpoint_status()
    
    print(f"\n📋 WHAT WOULD YOU LIKE TO DO?")
    
    if has_checkpoint:
        current_epoch = checkpoint_info.get('current_epoch', 1)
        best_acc = checkpoint_info.get('best_val_acc', 0)
        
        print(f"1. Continue training (currently at epoch {current_epoch}, best: {best_acc:.1f}%)")
        print(f"2. Test current model performance (full evaluation)")
        print(f"3. Quick inference test (10 sample predictions)")
        print(f"4. Start fresh training (will overwrite current progress)")
        print(f"5. Monitor training progress")
    else:
        print(f"1. Start new training")
        print(f"2. View system info")
    
    print(f"0. Exit")
    
    try:
        choice = input(f"\nEnter your choice: ").strip()
        
        if has_checkpoint:
            if choice == '1':
                current_epoch = checkpoint_info.get('current_epoch', 1)
                
                print(f"\n🎯 CONTINUE TRAINING")
                print(f"Current progress: Epoch {current_epoch}")
                print(f"\n💡 How many MORE epochs would you like to train?")
                print(f"   • 1 more epoch: ~0.4-0.9 hours")
                print(f"   • 5 more epochs: ~2-4.5 hours") 
                print(f"   • 10 more epochs: ~4-9 hours")
                print(f"   • 20+ more epochs: 9+ hours")
                
                try:
                    additional_epochs = int(input(f"\nHow many MORE epochs to train? "))
                    if additional_epochs >= 1:
                        target_epochs = current_epoch + additional_epochs - 1  # -1 because we're resuming
                        
                        print(f"\n📊 TRAINING PLAN:")
                        print(f"   Current: Epoch {current_epoch}")
                        print(f"   Additional: +{additional_epochs} epochs")
                        print(f"   New target: Epoch {target_epochs}")
                        
                        start_training(target_epochs, resume=True)
                    else:
                        print("❌ Must train at least 1 more epoch")
                        
                except ValueError:
                    print("❌ Invalid input - please enter a number")
                    
            elif choice == '2':
                test_model()
                
            elif choice == '3':
                quick_inference()
                
            elif choice == '4':
                confirm = input("⚠️ This will start fresh training (current progress will be lost). Continue? (y/N): ").strip().lower()
                if confirm == 'y':
                    print(f"\n🎯 FRESH START")
                    print(f"💡 Recommendation: Start with 1-3 epochs to test everything")
                    
                    try:
                        epochs = int(input("How many epochs to start with? "))
                        if 1 <= epochs <= 200:
                            start_training(epochs, resume=False)
                        else:
                            print("❌ Invalid number (must be 1-200)")
                    except ValueError:
                        print("❌ Invalid input - please enter a number")
                else:
                    print("❌ Cancelled")
                    
            elif choice == '5':
                os.system("cd /Users/piyushtyagi/FacialEmotion && python monitor_progress.py")
                
            elif choice == '0':
                print("👋 Goodbye!")
            else:
                print("❌ Invalid choice")
                
        else:
            if choice == '1':
                print(f"\n🎯 START FRESH TRAINING")
                print(f"💡 RECOMMENDATION: Start with just 1 epoch (~0.4-0.9 hours)")
                print(f"   This lets you:")
                print(f"   ✅ Verify everything works correctly")
                print(f"   ✅ See initial progress and accuracy")
                print(f"   ✅ Test the stop/resume functionality")
                print(f"   ✅ Then decide how many more epochs based on results")
                
                print(f"\n📊 EPOCH TIME GUIDE:")
                print(f"   • 1 epoch: ~0.4-0.9 hours (perfect for testing)")
                print(f"   • 3 epochs: ~1-3 hours (see real progress)")
                print(f"   • 5 epochs: ~2-4.5 hours (good initial results)")
                print(f"   • 10+ epochs: 4+ hours (serious training)")
                
                try:
                    epochs = int(input("\nHow many epochs to start with? (Recommend: 1-3): "))
                    if 1 <= epochs <= 200:
                        start_training(epochs, resume=False)
                    else:
                        print("❌ Invalid number (must be 1-200)")
                except ValueError:
                    print("❌ Invalid input - please enter a number")
                    
            elif choice == '2':
                print(f"\n📚 ULTRA-FLEXIBLE TRAINING SYSTEM:")
                print(f"🔄 Perfect Workflow:")
                print(f"   1. Start with 1 epoch (test everything works)")
                print(f"   2. Type 'stop' during training to save and exit")
                print(f"   3. Test your model performance")
                print(f"   4. Resume with more epochs based on:")
                print(f"      • How much time you have")
                print(f"      • How satisfied you are with current accuracy")
                print(f"   5. Repeat cycle: train → stop → test → resume")
                print(f"")
                print(f"🛡️  Safety Features:")
                print(f"   • Progress saved every 50 batches (2-3 minutes)")
                print(f"   • Resume from EXACT batch where you stopped")
                print(f"   • Never lose more than 2-3 minutes of work")
                print(f"   • Change epoch targets each resume session")
                print(f"")
                print(f"🎮 Controls During Training:")
                print(f"   • Type 'stop' and press Enter → Save and exit safely")
                print(f"   • Type 'quit' and press Enter → Save and exit safely")
                print(f"   • Never use Ctrl+C (not safe)")
                
            elif choice == '0':
                print("👋 Goodbye!")
            else:
                print("❌ Invalid choice")
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == '__main__':
    main()
