#!/bin/bash

# Quick Start Script for Emotion Detection Project
# This script will set up the environment and run basic tests

echo "🎭 EMOTION DETECTION - QUICK START"
echo "=================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.7+ first."
    exit 1
fi

# Check Python version
python3 -c "import sys; exit(0 if sys.version_info >= (3, 7) else 1)"
if [ $? -ne 0 ]; then
    echo "❌ Python 3.7+ is required. Please upgrade your Python version."
    exit 1
fi

echo "✅ Python version check passed"

# Install dependencies
echo "📦 Installing dependencies..."
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    echo "💡 Try running: pip install -r requirements.txt --user"
    exit 1
fi

echo "✅ Dependencies installed successfully"

# Run system tests
echo ""
echo "🧪 Running system tests..."
python3 test_system.py

if [ $? -ne 0 ]; then
    echo "❌ System tests failed"
    echo "💡 Please check the error messages above"
    exit 1
fi

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📖 Next steps:"
echo "1. Train a model: python3 emotion_detection.py"
echo "2. Run examples: python3 examples.py"
echo "3. Read the full documentation in README.md"
echo ""
echo "🎭 Happy emotion detection!"
