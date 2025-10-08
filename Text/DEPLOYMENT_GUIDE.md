# Streamlit Cloud Deployment Files
# =================================

## Essential Files for GitHub (Push these):

### Core Application
- app.py                      # Main Streamlit app
- emotion_detection.py        # ML model class
- requirements.txt            # Basic dependencies  
- requirements_streamlit.txt  # Streamlit dependencies

### Configuration & Setup
- config.py                   # Configuration settings
- utils.py                    # Utility functions
- .gitignore                  # Git ignore rules
- README.md                   # Documentation
- streamlit_config.toml       # Streamlit configuration

### Supporting Files
- examples.py                 # Usage examples
- setup.py                    # Setup script
- test_system.py             # System tests
- run_app.sh                 # Run script
- quick_start.sh             # Setup script

### Documentation
- confusion_matrix.png        # Performance chart (171KB)

## Files to EXCLUDE (too large or unnecessary):

### Model Files (419MB - exceeds GitHub limits)
- saved_emotion_model/        # Entire model directory
- model.safetensors          # 418MB model weights

### Generated/Temp Files  
- __pycache__/               # Python cache
- emotion_detection_env/     # Virtual environment
- emotion_model_results/     # Training artifacts
- batch_prediction_results.txt
- results/, plots/, models/  # Generated directories

## Deployment Strategy: Using Your Trained Model

### 🎯 RECOMMENDED: Deploy Your Custom Trained Model
Your `saved_emotion_model/` provides the best accuracy (93%+) and loads quickly once uploaded.

**Steps:**
1. **Train locally** (one-time, 20 minutes)
2. **Use Git LFS** for large model files  
3. **Push to GitHub** with model included
4. **Deploy to Streamlit Cloud** (loads in 5 seconds)

### 📋 Pre-Deployment Checklist:

## Complete Deployment Commands:

### Step 1: Train Model Locally (if not done)
```bash
# Train your custom model (20 minutes, one-time only)
python3 emotion_detection.py
# This creates saved_emotion_model/ directory with your trained model
```

### Step 2: Setup Git LFS for Large Files
```bash
# Install Git LFS (if not installed)
git lfs install

# Git LFS is already configured via .gitattributes
# Verify it's tracking your model files:
git lfs track
```

### Step 3: Add All Files Including Trained Model
```bash
# Add essential application files
git add app.py emotion_detection.py requirements.txt config.py
git add utils.py examples.py setup.py test_system.py
git add README.md .gitignore streamlit_config.toml DEPLOYMENT_GUIDE.md
git add .gitattributes confusion_matrix.png

# Add your trained model (Git LFS will handle the large files)
git add saved_emotion_model/

# Verify Git LFS is tracking large files
git lfs ls-files

# Commit everything
git commit -m "Deploy custom trained emotion detection model to Streamlit Cloud"

# Push to GitHub (Git LFS handles large files automatically)
git push origin main
```

### Step 4: Deploy to Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub repository  
3. Set main file: `app.py`
4. Deploy!
