#!/usr/bin/env python3
"""
Universal Emotion Classifier with Ollama Translation
===================================================

This system uses Ollama to translate Hindi/Hinglish to English, then uses
your existing English emotion detection model for classification.

Architecture:
Input Text → Language Detection → Translation (if needed) → English Emotion Model → Results

Features:
- Auto language detection
- Ollama translation for Hindi/Hinglish
- Caching for repeated texts
- Fallback mechanisms
- Performance monitoring
"""

import os
import re
import time
import json
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

try:
    import requests
except ImportError:
    print("⚠️  Warning: 'requests' not installed. Run: pip install requests")
    requests = None

# Import your existing emotion classifier
try:
    from emotion_detection import EmotionClassifier
    EMOTION_MODEL_AVAILABLE = True
except ImportError as e:
    EMOTION_MODEL_AVAILABLE = False
    IMPORT_ERROR = str(e)

@dataclass
class TranslationResult:
    """Result of translation operation."""
    original_text: str
    translated_text: str
    source_language: str
    translation_time: float
    success: bool
    error_message: Optional[str] = None

@dataclass
class EmotionResult:
    """Result of emotion prediction."""
    text: str
    original_text: str
    predicted_emotion: str
    confidence: float
    all_probabilities: Dict[str, float]
    translation_used: bool
    total_time: float
    translation_result: Optional[TranslationResult] = None

class LanguageDetector:
    """Simple language detection for Hindi/Hinglish vs English."""
    
    @staticmethod
    def detect_language(text: str) -> str:
        """
        Detect if text is English, Hindi, or Hinglish.
        
        Args:
            text: Input text to analyze
            
        Returns:
            'english', 'hindi', or 'hinglish'
        """
        text = text.lower().strip()
        
        # Devanagari script detection (Hindi)
        devanagari_chars = len(re.findall(r'[\u0900-\u097F]', text))
        
        # English alphabet detection
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        
        # Common Hindi/Hinglish words in Roman script
        hinglish_words = [
            'hai', 'hoon', 'hain', 'kya', 'kyun', 'kaise', 'kahan', 'yahan', 'wahan',
            'aur', 'par', 'lekin', 'kyunki', 'toh', 'phir', 'bhi', 'nahi', 'haan',
            'dil', 'pyaar', 'khushi', 'ghar', 'paani', 'khana', 'yaar', 'dost',
            'mummy', 'papa', 'beta', 'baccha', 'ladka', 'ladki', 'aadmi', 'aurat',
            'kaam', 'paisa', 'gadi', 'ghar', 'shaher', 'gaon', 'school', 'college',
            'achha', 'bura', 'sundar', 'pyara', 'mast', 'bindass', 'timepass',
            'bakwaas', 'kamaal', 'zabardast', 'ekdum', 'bilkul', 'thoda', 'bohot'
        ]
        
        # Count Hinglish words
        words = text.split()
        hinglish_word_count = sum(1 for word in words if word in hinglish_words)
        
        total_chars = len(text.replace(' ', ''))
        
        # Decision logic
        if total_chars == 0:
            return 'english'
        
        devanagari_ratio = devanagari_chars / total_chars
        
        # If significant Devanagari script, it's Hindi
        if devanagari_ratio > 0.3:
            return 'hindi'
        
        # If Hinglish words detected
        if hinglish_word_count > 0 or (hinglish_word_count / len(words) > 0.1 if words else False):
            return 'hinglish'
        
        # Default to English
        return 'english'

class OllamaTranslator:
    """Handles translation using Ollama."""
    
    def __init__(self, model_name: str = "qwen3-vl:8b-instruct", ollama_url: str = "http://localhost:11434"):
        """
        Initialize Ollama translator.
        
        Args:
            model_name: Ollama model to use for translation
            ollama_url: Ollama API endpoint
        """
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.api_url = f"{ollama_url}/api/generate"
        
        # Translation cache
        self.cache = {}
        
        print(f"🔄 Initializing Ollama Translator")
        print(f"🤖 Model: {model_name}")
        print(f"🌐 Endpoint: {ollama_url}")
        
        # Test connection
        if not self._test_connection():
            print("⚠️  Warning: Could not connect to Ollama. Translation will fail.")
    
    def _test_connection(self) -> bool:
        """Test if Ollama is running and model is available."""
        try:
            if not requests:
                print("❌ requests module not available")
                return False
                
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                if self.model_name in model_names:
                    print(f"✅ Ollama connected! Model {self.model_name} available.")
                    return True
                else:
                    print(f"❌ Model {self.model_name} not found. Available: {model_names}")
                    return False
            return False
        except Exception as e:
            print(f"❌ Ollama connection failed: {str(e)}")
            return False
    
    def translate_text(self, text: str, source_lang: str = "auto", max_retries: int = 3) -> TranslationResult:
        """
        Translate text using Ollama with retry logic.
        
        Args:
            text: Text to translate
            source_lang: Source language hint
            max_retries: Number of retry attempts
            
        Returns:
            TranslationResult object
        """
        start_time = time.time()
        
        # Check cache first
        cache_key = f"{text}_{source_lang}"
        if cache_key in self.cache:
            cached_result = self.cache[cache_key]
            print(f"💾 Using cached translation for: {text[:30]}...")
            return cached_result
        
        # For very short common phrases, use simple fallback
        simple_translations = {
            'bohot bhook lagi hai': 'I am very hungry',
            'bhook lagi hai': 'I am hungry',
            'pyaas lagi hai': 'I am thirsty',
            'neend aa rahi hai': 'I am feeling sleepy',
            'thak gaya hun': 'I am tired',
            'khush hun': 'I am happy',
            'gussa aa raha hai': 'I am getting angry',
            'dar lag raha hai': 'I am scared'
        }
        
        if text.lower().strip() in simple_translations:
            quick_translation = simple_translations[text.lower().strip()]
            print(f"🚀 Quick translation: {text} → {quick_translation}")
            
            result = TranslationResult(
                original_text=text,
                translated_text=quick_translation,
                source_language=source_lang,
                translation_time=0.01,
                success=True
            )
            self.cache[cache_key] = result
            return result
        
        # Prepare translation prompt
        if source_lang in ['hindi', 'hinglish']:
            prompt = f"""Translate to English briefly: {text}

English:"""
        else:
            prompt = f"""Translate to English: {text}

English:"""
        
        # Try translation with retries
        for attempt in range(max_retries):
            try:
                if not requests:
                    raise Exception("requests module not available")
                
                # Adjust timeout based on attempt
                timeout = 15 + (attempt * 10)  # 15s, 25s, 35s
                
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Lower for faster generation
                        "top_p": 0.9,
                        "max_tokens": 50    # Shorter for simple phrases
                    }
                }
                
                print(f"🔄 Translating (attempt {attempt + 1}/{max_retries}): {text[:50]}...")
                response = requests.post(self.api_url, json=payload, timeout=timeout)
                
                if response.status_code == 200:
                    result_data = response.json()
                    translated_text = result_data.get('response', '').strip()
                    
                    # Clean up translation
                    translated_text = self._clean_translation(translated_text)
                    
                    # Fallback if translation is empty or same as original
                    if not translated_text or translated_text.lower() == text.lower():
                        if 'bhook' in text.lower() or 'hungry' in text.lower():
                            translated_text = "I am hungry"
                        else:
                            translated_text = text  # Use original
                    
                    translation_time = time.time() - start_time
                    
                    result = TranslationResult(
                        original_text=text,
                        translated_text=translated_text,
                        source_language=source_lang,
                        translation_time=translation_time,
                        success=True
                    )
                    
                    # Cache the result
                    self.cache[cache_key] = result
                    
                    print(f"✅ Translation completed in {translation_time:.2f}s")
                    print(f"📝 Original: {text}")
                    print(f"🔄 Translated: {translated_text}")
                    
                    return result
                else:
                    print(f"⚠️  Ollama API error: {response.status_code} (attempt {attempt + 1})")
                    if attempt == max_retries - 1:  # Last attempt
                        error_msg = f"Ollama API error: {response.status_code}"
                        return self._create_fallback_result(text, source_lang, start_time, error_msg)
            
            except requests.exceptions.Timeout:
                print(f"⏰ Translation timeout (attempt {attempt + 1}/{max_retries})")
                if attempt == max_retries - 1:  # Last attempt
                    return self._create_fallback_result(text, source_lang, start_time, "Translation timeout")
            except Exception as e:
                print(f"❌ Translation error (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt == max_retries - 1:  # Last attempt
                    return self._create_fallback_result(text, source_lang, start_time, str(e))
        
        # Should not reach here, but just in case
        return self._create_fallback_result(text, source_lang, start_time, "All retries failed")
    
    def _create_fallback_result(self, text: str, source_lang: str, start_time: float, error_msg: str) -> TranslationResult:
        """Create a fallback translation result."""
        # Simple rule-based fallback for common Hinglish phrases
        fallback_text = text
        if 'bhook' in text.lower():
            fallback_text = "I am hungry"
        elif 'pyaas' in text.lower():
            fallback_text = "I am thirsty"
        elif 'khush' in text.lower():
            fallback_text = "I am happy"
        elif 'gussa' in text.lower():
            fallback_text = "I am angry"
        elif 'dar' in text.lower():
            fallback_text = "I am scared"
        elif 'pyaar' in text.lower():
            fallback_text = "I love"
        
        print(f"🔄 Using fallback translation: {fallback_text}")
        
        return TranslationResult(
            original_text=text,
            translated_text=fallback_text,
            source_language=source_lang,
            translation_time=time.time() - start_time,
            success=fallback_text != text,  # Success if we could translate
            error_message=error_msg
        )
    
    def _clean_translation(self, text: str) -> str:
        """Clean up translated text."""
        # Remove common artifacts
        text = re.sub(r'^(English translation:|Translation:|English:)', '', text, flags=re.IGNORECASE).strip()
        text = re.sub(r'^\W+', '', text)  # Remove leading punctuation
        
        # Basic sentence case
        if text and not text[0].isupper():
            text = text[0].upper() + text[1:]
        
        return text

class UniversalEmotionClassifier:
    """
    Universal emotion classifier that handles Hindi, Hinglish, and English.
    Uses Ollama for translation and existing English model for classification.
    """
    
    def __init__(self, ollama_model: str = "qwen3-vl:8b-instruct"):
        """
        Initialize the universal emotion classifier.
        
        Args:
            ollama_model: Ollama model to use for translation
        """
        self.ollama_model = ollama_model
        
        print("🚀 Initializing Universal Emotion Classifier")
        print("=" * 60)
        
        # Initialize components
        self.language_detector = LanguageDetector()
        self.translator = OllamaTranslator(model_name=ollama_model)
        
        # Initialize English emotion model
        self.english_model = None
        self._init_english_model()
        
        # Performance tracking
        self.stats = {
            'total_predictions': 0,
            'translations_used': 0,
            'successful_translations': 0,
            'average_time': 0.0
        }
        
        print("✅ Universal Emotion Classifier ready!")
        print("=" * 60)
    
    def _init_english_model(self):
        """Initialize the English emotion detection model."""
        try:
            if EMOTION_MODEL_AVAILABLE:
                self.english_model = EmotionClassifier()
                
                # Check if saved model exists, otherwise load base model
                saved_model_path = "./saved_emotion_model"
                if os.path.exists(saved_model_path):
                    try:
                        print("📥 Loading saved emotion model...")
                        self.english_model.load_saved_model(saved_model_path)
                        print("✅ Saved emotion model loaded successfully")
                    except Exception as e:
                        print(f"⚠️  Could not load saved model: {str(e)}")
                        print("📥 Loading base model for inference...")
                        self.english_model.load_model_and_tokenizer()
                        print("✅ Base model loaded for inference")
                else:
                    print("📥 Loading base model for inference...")
                    self.english_model.load_model_and_tokenizer()
                    print("✅ Base model loaded for inference")
            else:
                print(f"❌ Could not import EmotionClassifier: {IMPORT_ERROR}")
                print("🔧 Make sure emotion_detection.py is in the same directory")
                self.english_model = None
        except Exception as e:
            print(f"❌ Error loading English model: {str(e)}")
            self.english_model = None
    
    def predict_emotion(self, texts: Union[str, List[str]]) -> Union[EmotionResult, List[EmotionResult]]:
        """
        Predict emotions for given text(s).
        
        Args:
            texts: Single text string or list of texts
            
        Returns:
            EmotionResult or list of EmotionResults
        """
        # Handle single text
        if isinstance(texts, str):
            return self._predict_single(texts)
        
        # Handle multiple texts
        results = []
        for text in texts:
            results.append(self._predict_single(text))
        
        return results
    
    def _predict_single(self, text: str) -> EmotionResult:
        """Predict emotion for a single text."""
        start_time = time.time()
        
        print(f"\n🔍 Processing: {text[:50]}...")
        
        # Detect language
        detected_lang = self.language_detector.detect_language(text)
        print(f"🌍 Detected language: {detected_lang}")
        
        translation_result = None
        processed_text = text
        translation_used = False
        
        # Translate if needed
        if detected_lang in ['hindi', 'hinglish']:
            print(f"🔄 Translation needed for {detected_lang}")
            translation_result = self.translator.translate_text(text, detected_lang)
            
            if translation_result.success:
                processed_text = translation_result.translated_text
                translation_used = True
                self.stats['successful_translations'] += 1
            else:
                print("⚠️  Translation failed, using original text")
                processed_text = text
            
            self.stats['translations_used'] += 1
        else:
            print("✅ No translation needed (English detected)")
        
        # Get emotion prediction
        if self.english_model and self.english_model.model and self.english_model.tokenizer:
            try:
                emotion_prediction = self.english_model.predict_emotion([processed_text])
                
                if emotion_prediction:
                    pred_result = emotion_prediction[0]
                    predicted_emotion = pred_result['predicted_emotion']
                    confidence = pred_result['confidence']
                    all_probabilities = pred_result.get('all_probabilities', {})
                else:
                    # Fallback
                    predicted_emotion = 'neutral'
                    confidence = 0.5
                    all_probabilities = {'neutral': 0.5}
            
            except Exception as e:
                print(f"❌ Error in emotion prediction: {str(e)}")
                predicted_emotion = 'unknown'
                confidence = 0.0
                all_probabilities = {'unknown': 0.0}
        else:
            print("❌ English model not available or not properly loaded")
            predicted_emotion = 'unavailable'
            confidence = 0.0
            all_probabilities = {'unavailable': 0.0}
        
        total_time = time.time() - start_time
        
        # Update stats
        self.stats['total_predictions'] += 1
        self.stats['average_time'] = (
            (self.stats['average_time'] * (self.stats['total_predictions'] - 1) + total_time) /
            self.stats['total_predictions']
        )
        
        result = EmotionResult(
            text=processed_text,
            original_text=text,
            predicted_emotion=predicted_emotion,
            confidence=confidence,
            all_probabilities=all_probabilities,
            translation_used=translation_used,
            total_time=total_time,
            translation_result=translation_result
        )
        
        # Print results
        print(f"🎯 Emotion: {predicted_emotion} (confidence: {confidence:.3f})")
        print(f"⏱️  Total time: {total_time:.2f}s")
        if translation_used:
            print(f"🔄 Translation used: {translation_result.translation_time:.2f}s")
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            **self.stats,
            'translation_success_rate': (
                self.stats['successful_translations'] / max(1, self.stats['translations_used'])
            ),
            'model_available': self.english_model is not None
        }
    
    def batch_predict(self, texts: List[str], show_progress: bool = True) -> List[EmotionResult]:
        """
        Predict emotions for multiple texts with progress tracking.
        
        Args:
            texts: List of texts to process
            show_progress: Whether to show progress
            
        Returns:
            List of EmotionResult objects
        """
        results = []
        total_texts = len(texts)
        
        if show_progress:
            print(f"\n📊 Processing {total_texts} texts...")
        
        for i, text in enumerate(texts, 1):
            if show_progress:
                print(f"\n[{i}/{total_texts}] ", end="")
            
            result = self._predict_single(text)
            results.append(result)
            
            if show_progress and i % 5 == 0:
                stats = self.get_stats()
                print(f"\n📈 Progress: {i}/{total_texts} | Avg time: {stats['average_time']:.2f}s")
        
        return results

def interactive_demo(classifier: UniversalEmotionClassifier):
    """
    Interactive demo where users can input custom text for emotion prediction.
    """
    print("\n🎯 INTERACTIVE EMOTION DETECTION")
    print("=" * 60)
    print("📝 Enter your text in any language (English, Hindi, Hinglish)")
    print("💡 Examples:")
    print("   • English: 'I love spending time with my family'")
    print("   • Hindi: 'मुझे बहुत डर लग रहा है'")
    print("   • Hinglish: 'Yaar aj bohot khushi hui'")
    print("🚪 Type 'quit' or 'exit' to stop")
    print("-" * 60)
    
    while True:
        try:
            # Get user input
            user_text = input("\n📝 Enter your text: ").strip()
            
            # Check for exit conditions
            if user_text.lower() in ['quit', 'exit', 'q', 'bye']:
                print("\n👋 Thank you for using Universal Emotion Classifier!")
                break
            
            # Check for empty input
            if not user_text:
                print("⚠️  Please enter some text")
                continue
            
            # Predict emotion
            print("\n" + "🔄" * 20)
            result = classifier.predict_emotion(user_text)
            
            # Display beautiful results
            print("\n" + "🎭" * 20 + " RESULTS " + "🎭" * 20)
            print(f"📝 Original Text: \"{result.original_text}\"")
            
            if result.translation_used:
                print(f"🌍 Detected Language: {result.translation_result.source_language.title()}")
                print(f"🔄 Translated Text: \"{result.text}\"")
                print(f"⏱️  Translation Time: {result.translation_result.translation_time:.2f}s")
            else:
                print(f"🌍 Detected Language: English")
            
            print(f"🎯 Predicted Emotion: {result.predicted_emotion.upper()}")
            print(f"🎪 Confidence: {result.confidence:.1%}")
            
            # Show top emotions if available
            if result.all_probabilities and len(result.all_probabilities) > 1:
                print(f"\n📊 All Emotions:")
                sorted_emotions = sorted(result.all_probabilities.items(), 
                                       key=lambda x: x[1], reverse=True)
                for emotion, prob in sorted_emotions[:3]:  # Top 3
                    bar_length = int(prob * 20)  # 20 chars max
                    bar = "█" * bar_length + "░" * (20 - bar_length)
                    print(f"   {emotion.capitalize():<10} {bar} {prob:.1%}")
            
            print(f"\n⏱️  Total Processing Time: {result.total_time:.2f}s")
            print("🎭" * 50)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Thanks for using Universal Emotion Classifier!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {str(e)}")
            print("🔄 Please try again with different text")

def run_sample_tests(classifier: UniversalEmotionClassifier):
    """Run predefined sample tests."""
    print("\n🧪 SAMPLE TESTS")
    print("=" * 60)
    
    # Test texts
    test_texts = [
        "I am very happy today!",  # English
        "मैं बहुत खुश हूं आज!",  # Hindi
        "Yaar mujhe bahut gussa aa raha hai!",  # Hinglish
        "I feel so sad and lonely",  # English
        "Dil me bahut pyaar hai tumhare liye",  # Hinglish
    ]
    
    # Test individual predictions
    print("\n🔬 Individual Predictions:")
    for text in test_texts:
        result = classifier.predict_emotion(text)
        print("-" * 40)
    
    # Show final stats
    print("\n📊 Final Statistics:")
    stats = classifier.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")

# Example usage and testing
if __name__ == "__main__":
    print("🌟 UNIVERSAL EMOTION CLASSIFIER")
    print("=" * 60)
    print("🚀 Choose an option:")
    print("1️⃣  Interactive Mode (Enter your own text)")
    print("2️⃣  Sample Tests (Predefined examples)")
    print("3️⃣  Both (Run samples first, then interactive)")
    
    try:
        choice = input("\n📝 Enter your choice (1/2/3): ").strip()
        
        # Initialize classifier
        print(f"\n{'🔄' * 20} INITIALIZING {'�' * 20}")
        classifier = UniversalEmotionClassifier()
        
        if choice == "1":
            interactive_demo(classifier)
        elif choice == "2":
            run_sample_tests(classifier)
        elif choice == "3":
            run_sample_tests(classifier)
            interactive_demo(classifier)
        else:
            print("❌ Invalid choice. Running interactive mode...")
            interactive_demo(classifier)
            
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye! Thanks for using Universal Emotion Classifier!")
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        print("🔧 Please check your setup and try again")
