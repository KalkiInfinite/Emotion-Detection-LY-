# 🌟 Universal Emotion Classifier with Ollama Translation

A powerful multilingual emotion detection system that supports **English**, **Hindi**, and **Hinglish** text using Ollama for translation and transformer models for emotion classification.

## 🎯 Features

- **🌍 Multi-language Support**: English, Hindi, and Hinglish detection
- **🔄 Smart Translation**: Ollama-powered translation with fallback mechanisms
- **🎭 Six Emotions**: Joy, Sadness, Anger, Fear, Love, Surprise
- **💾 Caching System**: Fast repeated translations
- **📊 Interactive Interface**: Command-line and web-based testing
- **⚡ Real-time Processing**: Quick emotion prediction with confidence scores

## 🏗️ Architecture

```
Input Text → Language Detection → Translation (if needed) → English Emotion Model → Results
```

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+**
2. **Ollama** installed and running
3. **qwen3-vl:8b-instruct** model downloaded

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/universal-emotion-classifier.git
cd universal-emotion-classifier

# Create virtual environment
python -m venv emotion_env
source emotion_env/bin/activate  # On Windows: emotion_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download Ollama model (if not already done)
ollama pull qwen3-vl:8b-instruct
```

### Usage

#### 🎯 Interactive Mode
```bash
python universal_emotion_classifier.py
# Choose option 1 for interactive text input
```

#### 🧪 Quick Test
```bash
python quick_test.py
```

#### 🌐 Web Interface
```bash
streamlit run app_fast.py
```

## 📝 Example Usage

### English Text
```
📝 Input: "I'm so excited about my new job!"
🎯 Emotion: JOY (94.2%)
⏱️  Time: 0.05s
```

### Hindi Text
```
📝 Input: "मुझे बहुत डर लग रहा है"
🔄 Translation: "I am very scared"
🎯 Emotion: FEAR (96.8%)
⏱️  Time: 2.3s
```

### Hinglish Text
```
📝 Input: "Yaar bohot khushi ho rahi hai!"
🔄 Translation: "Friend, I'm feeling very happy!"
🎯 Emotion: JOY (89.5%)
⏱️  Time: 1.8s
```

## 🧠 Model Details

- **English Model**: BERT-based transformer (6 emotions)
- **Translation Model**: Ollama qwen3-vl:8b-instruct
- **Languages**: English, Hindi, Hinglish
- **Emotions**: Joy, Sadness, Anger, Fear, Love, Surprise

## 🔧 Configuration

### Ollama Setup
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve

# Download the translation model
ollama pull qwen3-vl:8b-instruct
```

### Environment Variables
```bash
export OLLAMA_HOST=localhost:11434  # Default Ollama endpoint
```

## 📊 Performance

- **Translation Speed**: ~2-5 seconds per text
- **Emotion Prediction**: ~0.05 seconds per text
- **Accuracy**: 85-95% depending on language and text complexity
- **Cache Hit Rate**: 100% for repeated texts

## 🎨 API Reference

### UniversalEmotionClassifier

```python
from universal_emotion_classifier import UniversalEmotionClassifier

# Initialize classifier
classifier = UniversalEmotionClassifier()

# Single prediction
result = classifier.predict_emotion("Your text here")
print(f"Emotion: {result.predicted_emotion}")
print(f"Confidence: {result.confidence:.2%}")

# Batch prediction
results = classifier.batch_predict([
    "I'm happy!",
    "मैं दुखी हूं",
    "Yaar bohot gussa aa raha hai"
])
```

### EmotionResult Structure

```python
@dataclass
class EmotionResult:
    text: str                    # Processed (translated) text
    original_text: str          # Original input text
    predicted_emotion: str      # Predicted emotion
    confidence: float          # Confidence score (0-1)
    all_probabilities: Dict    # All emotion probabilities
    translation_used: bool     # Whether translation was used
    total_time: float         # Total processing time
    translation_result: TranslationResult  # Translation details
```

## 🧪 Testing

### Test Samples
```python
# English samples
"I love spending time with my family"
"This traffic is so frustrating!"
"I'm really scared about tomorrow"

# Hindi samples  
"मुझे बहुत खुशी हो रही है"
"मैं बहुत डर गया हूं"
"मुझे गुस्सा आ रहा है"

# Hinglish samples
"Yaar maine job clear kar liya!"
"Bohot tension ho rahi hai"
"Dil me pyaar hai tumhare liye"
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ollama** for the excellent translation capabilities
- **Hugging Face Transformers** for the BERT model
- **dair-ai/emotion** dataset for training data

## 📞 Contact

- **Author**: Your Name
- **Email**: your.email@example.com
- **GitHub**: [@yourusername](https://github.com/yourusername)

---

**⭐ Star this repository if you found it helpful!**
