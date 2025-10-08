"""
Streamlit App for Text Emotion Detection
=========================================

A web interface for predicting emotions in text using fine-tuned BERT models.
Supports single text analysis, batch processing, and visualization of results.

Author: AI Assistant
Date: October 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import io
import base64
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Import the emotion classifier
try:
    from emotion_detection import EmotionClassifier
    MODEL_IMPORT_SUCCESS = True
except ImportError as e:
    MODEL_IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)

# Configure Streamlit page
st.set_page_config(
    page_title="Text Emotion Detection",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .emotion-result {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .confidence-text {
        font-size: 1.2rem;
        text-align: center;
        margin-top: 0.5rem;
    }
    .model-info {
        background-color: #6c757d;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .example-text {
        background-color: #e8f4fd;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
        cursor: pointer;
        border: 1px solid #d0e7ff;
    }
    .example-text:hover {
        background-color: #d0e7ff;
    }
</style>
""", unsafe_allow_html=True)

# Emotion color mapping
EMOTION_COLORS = {
    'joy': '#28a745',      # Green
    'love': '#e83e8c',     # Pink  
    'surprise': '#ffc107', # Yellow
    'anger': '#dc3545',    # Red
    'fear': '#6f42c1',     # Purple
    'sadness': '#6c757d'   # Gray
}

# Example texts for quick testing
EXAMPLE_TEXTS = [
    "I'm absolutely thrilled about this amazing opportunity! This is the best day ever!",
    "I'm really worried about the upcoming exam. What if I don't pass?",
    "You mean everything to me. I love spending time with you.",
    "This traffic is so frustrating! Why does this always happen when I'm late?",
    "I feel so empty and lost. Nothing seems to matter anymore.",
    "Wow! I never expected to see you here! What a pleasant surprise!"
]

@st.cache_resource
def load_emotion_model():
    """Load the emotion detection model with caching."""
    try:
        if not MODEL_IMPORT_SUCCESS:
            return None, f"Failed to import EmotionClassifier: {IMPORT_ERROR}"
        
        model_path = './saved_emotion_model/'
        if not os.path.exists(model_path):
            return None, f"Trained model not found at: {model_path}. Please train the model first using: python3 emotion_detection.py"
        
        classifier = EmotionClassifier()
        classifier.load_saved_model(model_path)
        return classifier, "Custom trained model loaded successfully!"
    
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

def create_emotion_chart(probabilities: Dict[str, float]) -> go.Figure:
    """Create an interactive bar chart for emotion probabilities."""
    emotions = list(probabilities.keys())
    probs = list(probabilities.values())
    colors = [EMOTION_COLORS.get(emotion, '#1f77b4') for emotion in emotions]
    
    fig = go.Figure(data=[
        go.Bar(
            x=emotions,
            y=probs,
            marker_color=colors,
            text=[f"{p:.1%}" for p in probs],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Probability: %{y:.1%}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title="Emotion Probability Distribution",
        xaxis_title="Emotions",
        yaxis_title="Probability",
        yaxis=dict(tickformat=".0%"),
        height=400,
        template="plotly_white"
    )
    
    return fig

def create_confidence_gauge(confidence: float, emotion: str) -> go.Figure:
    """Create a gauge chart for confidence level."""
    color = EMOTION_COLORS.get(emotion, '#1f77b4')
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Confidence Level<br><span style='font-size:0.8em;color:gray'>Predicted: {emotion.title()}</span>"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 25], 'color': "lightgray"},
                {'range': [25, 50], 'color': "gray"},
                {'range': [50, 75], 'color': "lightblue"},
                {'range': [75, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def process_batch_predictions(classifier, df: pd.DataFrame) -> pd.DataFrame:
    """Process batch predictions for uploaded CSV file."""
    if 'text' not in df.columns:
        st.error("CSV file must contain a 'text' column!")
        return None
    
    # Get predictions for all texts
    texts = df['text'].astype(str).tolist()
    
    with st.spinner("Analyzing emotions for all texts..."):
        # Suppress the print statements from predict_emotion
        import contextlib
        from io import StringIO
        
        f = StringIO()
        with contextlib.redirect_stdout(f):
            results = classifier.predict_emotion(texts)
    
    # Create results dataframe
    results_data = []
    for i, result in enumerate(results):
        results_data.append({
            'Text': result['text'][:100] + "..." if len(result['text']) > 100 else result['text'],
            'Predicted_Emotion': result['predicted_emotion'].title(),
            'Confidence': f"{result['confidence']:.1%}",
            'Top_3_Predictions': " | ".join([f"{emotion}: {prob:.1%}" for emotion, prob in result['top_predictions']])
        })
    
    return pd.DataFrame(results_data)

def main():
    """Main Streamlit app function."""
    
    # Header
    st.markdown('<h1 class="main-header">🎭 Text Emotion Detection</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load model
    classifier, load_message = load_emotion_model()
    
    # Sidebar - Model Information
    with st.sidebar:
        st.header("📊 Model Information")
        
        if classifier is not None:
            st.markdown("""
            <div class="model-info">
                <h4>✅ Model Status: Ready</h4>
                <ul>
                    <li><b>Architecture:</b> BERT-base-uncased</li>
                    <li><b>Parameters:</b> ~110M</li>
                    <li><b>Expected Accuracy:</b> 92.75%</li>
                    <li><b>Training Data:</b> 20K emotion samples</li>
                    <li><b>Emotions:</b> 6 classes</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.subheader("🎯 Emotion Classes")
            emotions = ['Joy', 'Sadness', 'Love', 'Anger', 'Fear', 'Surprise']
            for emotion in emotions:
                color = EMOTION_COLORS.get(emotion.lower(), '#1f77b4')
                st.markdown(f"<span style='color: {color}; font-weight: bold;'>● {emotion}</span>", unsafe_allow_html=True)
        else:
            st.error(f"❌ Model Loading Failed")
            st.error(load_message)
            st.markdown("""
            **To fix this issue:**
            1. Ensure `emotion_detection.py` is in the same directory
            2. Train the model first: `python emotion_detection.py`
            3. Check that `./saved_emotion_model/` directory exists
            """)
    
    # Main content
    if classifier is None:
        st.error("⚠️ Cannot proceed without a loaded model. Please check the sidebar for details.")
        return
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["📝 Single Text Analysis", "📊 Batch Analysis", "🔬 Model Details"])
    
    with tab1:
        st.header("Analyze Single Text")
        
        # Example texts section
        st.subheader("💡 Quick Examples (Click to use)")
        cols = st.columns(2)
        
        for i, example in enumerate(EXAMPLE_TEXTS):
            col = cols[i % 2]
            if col.button(f"Example {i+1}: {example[:50]}...", key=f"example_{i}"):
                st.session_state.text_input = example
                st.rerun()
        
        # Text input area
        user_text = st.text_area(
            "Enter your text for emotion analysis:",
            value=st.session_state.get('text_input', ''),
            height=150,
            placeholder="Type or paste your text here... (e.g., 'I'm so excited about this new project!')",
            key="text_input"
        )
        
        # Analysis button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_button = st.button("🔍 Analyze Emotion", type="primary", use_container_width=True)
        
        # Process single text analysis
        if analyze_button and user_text.strip():
            with st.spinner("Analyzing emotion..."):
                # Suppress print statements
                import contextlib
                from io import StringIO
                
                f = StringIO()
                with contextlib.redirect_stdout(f):
                    results = classifier.predict_emotion([user_text])
                
                result = results[0]
                emotion = result['predicted_emotion']
                confidence = result['confidence']
                all_probs = result['all_probabilities']
            
            # Display results
            st.success("✅ Analysis Complete!")
            
            # Main emotion result
            emotion_color = EMOTION_COLORS.get(emotion, '#1f77b4')
            st.markdown(f"""
            <div class="emotion-result" style="background-color: {emotion_color}20; border: 2px solid {emotion_color};">
                <span style="color: {emotion_color};">🎭 {emotion.upper()}</span>
                <div class="confidence-text">Confidence: {confidence:.1%}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Charts in columns
            col1, col2 = st.columns(2)
            
            with col1:
                # Confidence gauge
                gauge_fig = create_confidence_gauge(confidence, emotion)
                st.plotly_chart(gauge_fig, use_container_width=True)
            
            with col2:
                # Probability chart
                prob_fig = create_emotion_chart(all_probs)
                st.plotly_chart(prob_fig, use_container_width=True)
            
            # Detailed results
            with st.expander("📈 Detailed Probabilities"):
                prob_df = pd.DataFrame([
                    {"Emotion": emotion.title(), "Probability": f"{prob:.3f}", "Percentage": f"{prob:.1%}"}
                    for emotion, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
                ])
                st.dataframe(prob_df, use_container_width=True)
        
        elif analyze_button and not user_text.strip():
            st.warning("⚠️ Please enter some text to analyze.")
    
    with tab2:
        st.header("Batch Analysis")
        st.markdown("Upload a CSV file with a 'text' column to analyze multiple texts at once.")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="CSV file must contain a column named 'text'"
        )
        
        if uploaded_file is not None:
            try:
                # Read CSV
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ File uploaded successfully! Found {len(df)} rows.")
                
                # Show preview
                with st.expander("📋 Data Preview"):
                    st.dataframe(df.head(), use_container_width=True)
                
                # Process batch predictions
                if st.button("🔍 Analyze All Texts", type="primary"):
                    results_df = process_batch_predictions(classifier, df)
                    
                    if results_df is not None:
                        st.success("✅ Batch analysis complete!")
                        
                        # Display results
                        st.subheader("📊 Results")
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Summary statistics
                        emotion_counts = results_df['Predicted_Emotion'].value_counts()
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("📈 Emotion Distribution")
                            fig_pie = px.pie(
                                values=emotion_counts.values,
                                names=emotion_counts.index,
                                color=emotion_counts.index,
                                color_discrete_map={emotion.title(): color for emotion, color in EMOTION_COLORS.items()}
                            )
                            st.plotly_chart(fig_pie, use_container_width=True)
                        
                        with col2:
                            st.subheader("📊 Summary Statistics")
                            st.metric("Total Texts", len(results_df))
                            st.metric("Most Common Emotion", emotion_counts.index[0])
                            st.metric("Unique Emotions", len(emotion_counts))
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name="emotion_analysis_results.csv",
                            mime="text/csv"
                        )
            
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
        
        # Sample CSV download
        st.subheader("📄 Sample CSV Format")
        sample_df = pd.DataFrame({
            'text': [
                "I'm so happy today!",
                "This is really frustrating.",
                "I love spending time with family.",
                "I'm worried about the results."
            ]
        })
        
        st.dataframe(sample_df, use_container_width=True)
        
        csv_sample = sample_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Sample CSV",
            data=csv_sample,
            file_name="sample_emotion_texts.csv",
            mime="text/csv"
        )
    
    with tab3:
        st.header("🔬 Model Technical Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🤖 Architecture Details")
            st.markdown("""
            - **Model Type**: BERT (Bidirectional Encoder Representations from Transformers)
            - **Variant**: bert-base-uncased
            - **Parameters**: ~110 million
            - **Layers**: 12 transformer layers
            - **Hidden Size**: 768 dimensions
            - **Attention Heads**: 12 per layer
            - **Max Sequence Length**: 128 tokens
            """)
            
            st.subheader("🎯 Training Details")
            st.markdown("""
            - **Training Data**: dair-ai/emotion dataset
            - **Training Samples**: 16,000
            - **Validation Samples**: 2,000  
            - **Test Samples**: 2,000
            - **Training Epochs**: 3
            - **Learning Rate**: 2e-5
            - **Batch Size**: 16
            """)
        
        with col2:
            st.subheader("📈 Performance Metrics")
            st.markdown("""
            - **Test Accuracy**: 92.75%
            - **Test F1-Score**: 92.72%
            - **Training Time**: ~60 minutes (GPU)
            """)
            
            st.subheader("🎭 Per-Emotion Performance")
            performance_data = {
                'Emotion': ['Sadness', 'Joy', 'Anger', 'Fear', 'Love', 'Surprise'],
                'F1-Score': [96.6, 94.8, 91.5, 89.9, 83.5, 73.4],
                'Performance': ['Excellent', 'Excellent', 'Very Good', 'Very Good', 'Good', 'Acceptable']
            }
            perf_df = pd.DataFrame(performance_data)
            st.dataframe(perf_df, use_container_width=True)
        
        st.subheader("🔧 Technical Implementation")
        st.markdown("""
        **Preprocessing:**
        - Tokenization using BERT WordPiece tokenizer
        - Sequence truncation/padding to 128 tokens
        - Special tokens: [CLS] for classification, [SEP] for separation
        
        **Model Architecture:**
        - Pre-trained BERT encoder (12 layers)
        - Classification head: Linear layer (768 → 6)
        - Activation: GELU in encoder, Softmax for final probabilities
        
        **Device Support:**
        - CUDA GPUs (with mixed precision FP16)
        - Apple Silicon MPS acceleration  
        - CPU fallback
        """)

if __name__ == "__main__":
    main()
