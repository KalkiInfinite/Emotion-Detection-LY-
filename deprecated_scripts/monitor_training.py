"""
Monitor MPS training progress and system resources.
"""

import time
import os
import json
from pathlib import Path
import subprocess

def monitor_training():
    """Monitor the training progress."""
    print("🔍 MPS Training Monitor")
    print("=" * 50)
    
    # Check if training is running
    result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
    if 'train_mps.py' in result.stdout:
        print("✅ Training is running!")
    else:
        print("❌ Training not detected")
        return
    
    # Monitor files
    log_dir = Path('./outputs/mps_training')
    log_file = log_dir / 'training.log'
    
    if log_file.exists():
        print(f"\n📊 Latest training logs:")
        print("-" * 30)
        
        # Read last few lines
        with open(log_file, 'r') as f:
            lines = f.readlines()
            for line in lines[-10:]:  # Last 10 lines
                print(line.strip())
    else:
        print("⏳ Log file not created yet...")
    
    # Check system resources (simplified)
    print(f"\n⚡ System Status:")
    print("-" * 20)
    
    # Memory usage
    result = subprocess.run(['vm_stat'], capture_output=True, text=True)
    if result.returncode == 0:
        lines = result.stdout.split('\n')
        for line in lines[:3]:  # First few lines with page info
            if 'Pages' in line:
                print(f"   {line.strip()}")
    
    # Check GPU activity (if available)
    result = subprocess.run(['sudo', 'powermetrics', '--samplers', 'gpu_power', '-n', '1', '-i', '1000'], 
                          capture_output=True, text=True)
    if result.returncode == 0 and 'GPU' in result.stdout:
        print("   🎮 GPU is active")
    
    print(f"\n⏰ Monitoring at {time.strftime('%H:%M:%S')}")

if __name__ == '__main__':
    try:
        while True:
            monitor_training()
            print("\n" + "="*50)
            time.sleep(30)  # Check every 30 seconds
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped")
