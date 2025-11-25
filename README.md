# Vision Transformer for Facial Emotion Recognition

A comprehensive implementation of facial emotion recognition using Vision Transformer (ViT) architecture with PyTorch. This project supports training, evaluation, and inference on multiple emotion datasets.

## 🚀 Features

- **State-of-the-art ViT Architecture**: Uses pretrained Vision Transformer from Hugging Face
- **Multiple Dataset Support**: FER2013, AffectNet, RAF-DB, and custom directory datasets
- **Comprehensive Training Pipeline**: With data augmentation, balanced sampling, and tensorboard logging
- **Detailed Evaluation**: Confusion matrices, per-class metrics, and visualization
- **Real-time Inference**: Single image, batch processing, and webcam support
- **Modular Design**: Easy to extend and customize

## 📁 Project Structure

```
FacialEmotion/
├── README.md
├── requirements.txt
├── train.py                    # Training script
├── eval.py                     # Evaluation script
├── inference.py               # Inference script
├── config/
│   └── emotion_config.py      # Configuration settings
├── data/
│   ├── __init__.py
│   ├── dataloader.py          # Dataset classes and data loaders
│   └── transforms.py          # Data augmentation and preprocessing
├── models/
│   ├── __init__.py
│   └── vit_emotion_model.py   # ViT emotion recognition model
├── notebooks/                 # Jupyter notebooks for analysis
├── outputs/
│   ├── checkpoints/           # Saved model checkpoints
│   ├── logs/                  # Training logs and tensorboard files
│   └── evaluation/            # Evaluation results and plots
```

## ⚡ Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd FacialEmotion

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Your Dataset

**Option A: Directory Structure (Recommended)**
```
your_dataset/
├── angry/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── happy/
│   ├── image1.jpg
│   └── ...
└── ... (other emotion folders)
```

**Option B: FER2013 Format**
- Download FER2013 dataset
- Extract images to `data/fer2013/images/`
- Place `fer2013.csv` in `data/fer2013/`

### 3. Training

```bash
# Train with directory dataset
python train.py --data_dir /path/to/your/dataset --dataset_type directory

# Train with FER2013
python train.py --data_dir /path/to/fer2013 --dataset_type fer2013

# Custom training parameters
python train.py \
    --data_dir /path/to/dataset \
    --batch_size 64 \
    --num_epochs 100 \
    --learning_rate 1e-4 \
    --model_name google/vit-large-patch16-224
```

### 4. Evaluation

```bash
# Evaluate trained model
python eval.py \
    --checkpoint outputs/checkpoints/best_model.pt \
    --data_dir /path/to/test/dataset \
    --dataset_type directory
```

### 5. Inference

```bash
# Single image prediction
python inference.py \
    --checkpoint outputs/checkpoints/best_model.pt \
    --mode single \
    --input path/to/image.jpg

# Batch processing
python inference.py \
    --checkpoint outputs/checkpoints/best_model.pt \
    --mode batch \
    --input /path/to/image/directory \
    --output results.json

# Real-time webcam inference
python inference.py \
    --checkpoint outputs/checkpoints/best_model.pt \
    --mode webcam
```

## 🔧 Configuration

Modify `config/emotion_config.py` to customize:

- **Model settings**: Architecture, dropout, freeze options
- **Training parameters**: Learning rate, batch size, epochs
- **Data augmentation**: Flip, rotation, color jittering
- **Hardware settings**: GPU usage, mixed precision

## 📊 Dataset Details

### FER2013
- **Classes**: 7 (angry, disgust, fear, happy, sad, surprise, neutral)
- **Images**: ~35,000 grayscale 48x48 images
- **Features**: Grayscale images, balanced class distribution
- **Layout**: Organized into `train`, `test`, and `validation` splits
- **Preprocessing**: Resized to 224x224, normalized pixel values
- **Postprocessing**: None

### AffectNet
- **Classes**: 8 (neutral, happiness, sadness, surprise, fear, disgust, anger, contempt)
- **Images**: 400,000+ color images
- **Features**: High-resolution color images, imbalanced class distribution
- **Layout**: Organized into `train` and `validation` splits
- **Preprocessing**: Resized to 224x224, normalized pixel values
- **Postprocessing**: None

### RAF-DB
- **Classes**: 7 (surprise, fear, disgust, happiness, sadness, anger, neutral)
- **Images**: 30,000 color images
- **Features**: Color images, balanced class distribution
- **Layout**: Organized into `train` and `test` splits
- **Preprocessing**: Resized to 224x224, normalized pixel values
- **Postprocessing**: None

### Custom Dataset
- **Classes**: User-defined
- **Images**: User-provided
- **Features**: Flexible format, supports grayscale and color images
- **Layout**: Emotion-labeled directories
- **Preprocessing**: Resized to 224x224, normalized pixel values
- **Postprocessing**: None

## 🧠 Methodology

1. **Data Collection**: Utilized FER2013, AffectNet, RAF-DB, and custom datasets.
2. **Preprocessing**: Resized images to 224x224, applied normalization, and data augmentation (rotation, flip, color jittering).
3. **Model Architecture**:
   - **Backbone**: Vision Transformer `google/vit-base-patch16-224`.
   - **Classifier Head**: LayerNorm, Dropout (0.1), Linear layers.
   - **Parameters**: ~86.7M total, ~300K in the classifier head.
4. **Training**:
   - **Optimizer**: AdamW with weight decay.
   - **Scheduler**: Cosine annealing with warmup.
   - **Loss Function**: Cross-entropy loss.
   - **Batch Size**: 32.
   - **Epochs**: 50.
   - **Device**: Apple Silicon MPS.
5. **Evaluation**:
   - Metrics: Accuracy, precision, recall, F1-score.
   - Tools: Confusion matrix, classification report.
6. **Inference**:
   - Modes: Single image, batch processing, real-time webcam.

## 📊 Results

### Overall Metrics
- **Accuracy**: 69.24%
- **Precision**: 62.48%
- **Recall**: 59.56%
- **F1-Score**: 59.80%

### Per-Class Metrics
| Emotion   | Precision | Recall | F1-Score | Accuracy |
|-----------|-----------|--------|----------|----------|
| Angry     | 62.46%    | 59.08% | 60.72%   | 59.08%   |
| Contempt  | 50.88%    | 39.46% | 44.44%   | 39.46%   |
| Disgust   | 48.53%    | 46.92% | 47.71%   | 46.92%   |
| Fear      | 60.39%    | 24.52% | 34.88%   | 24.52%   |
| Happy     | 89.16%    | 87.21% | 88.18%   | 87.21%   |
| Neutral   | 67.76%    | 73.39% | 70.46%   | 73.39%   |
| Sad       | 52.15%    | 60.72% | 56.11%   | 60.72%   |
| Surprise  | 68.52%    | 85.15% | 75.93%   | 85.15%   |

### Training Details
- **Dropout**: 0.1
- **Layers**: 12 transformer layers, 12 attention heads
- **Classes**: 7 (FER2013), 8 (AffectNet), 7 (RAF-DB)

### Outputs
- **Confusion Matrix**: Visualized in `outputs/evaluation/confusion_matrix.png`
- **Class Distribution**: Visualized in `outputs/evaluation/class_distribution.png`
- **Evaluation Report**: JSON file in `outputs/evaluation/evaluation_results.json`

## 🎯 Model Architecture

The model uses a pretrained Vision Transformer as backbone:

1. **Backbone**: Pretrained ViT (google/vit-base-patch16-224)
2. **Image Processing**: 224x224 patches with positional embedding
3. **Feature Extraction**: Multi-head self-attention layers
4. **Classification Head**: LayerNorm + Dropout + Linear layers
5. **Output**: Emotion class probabilities

## 📈 Training Features

- **Data Augmentation**: Rotation, flip, color jittering, Gaussian noise
- **Balanced Sampling**: Handles class imbalance automatically
- **Mixed Precision**: Faster training with reduced memory usage
- **Tensorboard Logging**: Real-time monitoring of metrics
- **Checkpointing**: Automatic saving of best models
- **Learning Rate Scheduling**: Cosine annealing with warmup

## 🔍 Evaluation Metrics

The evaluation provides comprehensive analysis:

- **Overall Metrics**: Accuracy, macro/weighted F1-score
- **Per-class Metrics**: Precision, recall, F1-score for each emotion
- **Confusion Matrix**: Both raw counts and normalized versions
- **Visualizations**: Class distribution, performance plots
- **Classification Report**: Detailed scikit-learn report

## 🎥 Real-time Inference

The webcam mode provides:

- **Face Detection**: Automatic face detection using OpenCV
- **Live Predictions**: Real-time emotion classification
- **Probability Display**: Visual probability bars for all emotions
- **Video Recording**: Optional video saving functionality

## ⚙️ Advanced Usage

### Custom Model Configuration

```python
from models.vit_emotion_model import ViTEmotionClassifier

model = ViTEmotionClassifier(
    model_name="google/vit-large-patch16-224",
    num_classes=7,
    dropout_rate=0.2,
    freeze_backbone=False
)
```

### Test-Time Augmentation

```bash
python inference.py \
    --checkpoint best_model.pt \
    --mode single \
    --input image.jpg \
    --use_tta
```

### Multi-GPU Training

```python
# In your training script
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.25+
- OpenCV 4.6+
- scikit-learn 1.0+
- See `requirements.txt` for complete list

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@misc{vit-emotion-recognition,
  title={Vision Transformer for Facial Emotion Recognition},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/FacialEmotion}
}
```

## 🙏 Acknowledgments

- Hugging Face for pretrained Vision Transformer models
- PyTorch team for the deep learning framework
- OpenCV community for computer vision tools
- Emotion dataset creators (FER2013, AffectNet, RAF-DB)

## 📞 Support

For questions and support:
- Open an issue on GitHub
- Check the documentation in `/docs`
- Review example notebooks in `/notebooks`

---

**Happy emotion recognition! 😊**
