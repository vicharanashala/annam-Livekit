FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

# Set working directory
WORKDIR /app

# HuggingFace token for gated model access
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Flash Attention 2 using pre-compiled wheel (PyTorch 2.2, CUDA 12.x, Python 3.10)
RUN pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.6/flash_attn-2.5.6+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install huggingface_hub for login
RUN pip install huggingface_hub

# Pre-download model to cache in image (with HF token)
RUN python -c "from huggingface_hub import login; login(token='${HF_TOKEN}'); \
    from parler_tts import ParlerTTSForConditionalGeneration; from transformers import AutoTokenizer; \
    model = ParlerTTSForConditionalGeneration.from_pretrained('ai4bharat/indic-parler-tts', token='${HF_TOKEN}'); \
    tokenizer = AutoTokenizer.from_pretrained('ai4bharat/indic-parler-tts', token='${HF_TOKEN}'); \
    desc_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path, token='${HF_TOKEN}')"

# Copy application code
COPY main.py .

# Expose port
EXPOSE 8890

# Run the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8890"]
