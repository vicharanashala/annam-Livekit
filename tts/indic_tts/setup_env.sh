#!/bin/bash
# Setup script for Indic TTS environment

set -e

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support
pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install requirements
pip install -r requirements.txt

# Create models directory
mkdir -p models

echo "================================"
echo "Environment setup complete!"
echo "Activate with: source venv/bin/activate"
echo "================================"
echo ""
echo "Next steps:"
echo "1. Download model weights from:"
echo "   https://github.com/AI4Bharat/Indic-TTS/releases/tag/v1-checkpoints-release"
echo "2. Extract to the 'models' directory"
echo "3. Run: python inference.py"
