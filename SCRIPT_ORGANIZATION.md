# 🎯 **FacialEmotion Project - Script Organization Guide**

## ✅ **CURRENTLY USEFUL SCRIPTS (Keep and Use)**

### **🚀 Main Training Scripts:**
- **`flexible_training.py`** ⭐ **YOUR MAIN SCRIPT** 
  - Ultra-flexible training with user-defined epochs
  - Stop anytime, resume with different targets
  - Perfect for your workflow
  - **Use this for all training!**

- **`train_safe.py`** 🛡️ **Core Training Engine**
  - Batch-level checkpointing system
  - MPS optimization for Apple Silicon
  - Called by flexible_training.py
  - **Don't run directly - use flexible_training.py**

### **📊 Evaluation & Testing:**
- **`eval.py`** 🧪 **Model Evaluation**
  - Full test set evaluation
  - Confusion matrix, per-class metrics
  - Called by flexible_training.py

- **`inference.py`** 🎯 **Quick Predictions**
  - Test model on sample images
  - Quick inference testing
  - Called by flexible_training.py

### **📈 Monitoring:**
- **`monitor_progress.py`** 📊 **Training Monitor**
  - Real-time progress tracking
  - Training plots and statistics
  - Use while training is running

### **🎮 Alternative Launchers (Optional):**
- **`resume_training.py`** 🔄 **Simple Resume**
  - Basic resume functionality
  - Less flexible than flexible_training.py
  - **Use flexible_training.py instead**

- **`train_options.py`** ⚡ **Quick Options**
  - Pre-defined epoch options (15, 50)
  - Less flexible than flexible_training.py
  - **Use flexible_training.py instead**

### **📁 Data Management:**
- **`dataset_manager.py`** 📦 **Dataset Organization**
  - Already used to organize your 121K images
  - **No need to run again unless adding new datasets**

## ❌ **DEPRECATED SCRIPTS (Moved to deprecated_scripts/)**

### **Old Training Scripts:**
- **`train.py`** - Basic training, replaced by train_safe.py
- **`train_mps.py`** - MPS training, features merged into train_safe.py
- **`train_fast.py`** - Fast training, replaced by flexible system
- **`train_unified.py`** - Unified training, replaced by flexible system

### **Setup & Testing Scripts:**
- **`setup_dataset.py`** - Dataset setup, already done
- **`quick_test.py`** - Quick testing, replaced by flexible_training.py
- **`test_setup.py`** - Setup testing, no longer needed
- **`organize_datasets.py`** - Dataset organization, replaced by dataset_manager.py
- **`monitor_training.py`** - Old monitoring, replaced by monitor_progress.py

---

## 🎯 **YOUR SIMPLIFIED WORKFLOW:**

### **For Training:**
```bash
python flexible_training.py
```
**This is your ONE main script for all training needs!**

### **For Monitoring (separate terminal):**
```bash
python monitor_progress.py
```

### **Your Core Files Structure:**
```
📁 FacialEmotion/
├── 🚀 flexible_training.py    ← YOUR MAIN SCRIPT
├── 🛡️ train_safe.py           ← Core engine (don't run directly)
├── 🧪 eval.py                 ← Evaluation (auto-called)
├── 🎯 inference.py            ← Quick predictions (auto-called)
├── 📊 monitor_progress.py     ← Progress monitoring
├── 📦 dataset_manager.py      ← Dataset management (already used)
├── 📁 models/                 ← Your ViT model
├── 📁 data/                   ← Data loading code
├── 📁 outputs/                ← Training results & checkpoints
├── 📁 deprecated_scripts/     ← Old unused scripts
└── 📁 data/unified_dataset/   ← Your 121K images dataset
```

## 💡 **What You Need to Remember:**

1. **Use `flexible_training.py` for ALL training** - it's your one-stop solution
2. **Monitor with `monitor_progress.py`** while training runs
3. **Everything else is automated** - eval.py and inference.py are called automatically
4. **Deprecated scripts are safely stored** but not needed

## 🎮 **Quick Commands:**
- **Start training:** `python flexible_training.py`
- **Monitor progress:** `python monitor_progress.py` 
- **That's it!** Everything else is handled automatically.
