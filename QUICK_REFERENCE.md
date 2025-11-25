# 🎯 **QUICK REFERENCE - What to Use**

## ⭐ **YOUR MAIN SCRIPT:**
```bash
python flexible_training.py
```
**Use this for ALL training!** It handles everything:
- Start with any number of epochs
- Stop anytime safely 
- Resume with different targets
- Automatic testing options

## 📊 **MONITORING (run in separate terminal):**
```bash
python monitor_progress.py
```

## 📁 **USEFUL SCRIPTS BREAKDOWN:**

### **🚀 TRAINING:**
- **`flexible_training.py`** ← **YOUR MAIN SCRIPT** (use this!)
- **`train_safe.py`** ← Core engine (don't run directly)

### **🧪 EVALUATION:**
- **`eval.py`** ← Auto-called by flexible_training.py
- **`inference.py`** ← Auto-called by flexible_training.py

### **📈 MONITORING:**
- **`monitor_progress.py`** ← Use while training

### **🎮 ALTERNATIVES (less flexible):**
- **`resume_training.py`** ← Basic resume (use flexible_training.py instead)
- **`train_options.py`** ← Fixed options (use flexible_training.py instead)

### **📦 DATA:**
- **`dataset_manager.py`** ← Already used (don't need to run again)

## ❌ **IGNORE THESE:**
- **`deprecated_scripts/`** folder - old scripts, safely stored but not needed

## 💡 **REMEMBER:**
- **One main script:** `flexible_training.py`
- **One monitoring script:** `monitor_progress.py`
- **Everything else is automatic!**
