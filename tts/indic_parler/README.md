# Indic Parler TTS - OpenAI-Compatible API

FastAPI server for [AI4Bharat's Indic Parler TTS](https://huggingface.co/ai4bharat/indic-parler-tts) with OpenAI-style API.

## Features

- **21+ Indian Languages**: Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam, and more
- **69 Unique Voices**: Pre-configured voice descriptions for consistent output
- **OpenAI-Compatible**: Drop-in replacement for OpenAI TTS API

## Quick Start

### Using Docker (Recommended)

```bash
docker-compose up --build
```

### Manual Run

```bash
pip install -r requirements.txt
python main.py
```

The server starts at `http://localhost:8890`

## API Endpoints

### Generate Speech
```bash
curl -X POST http://localhost:8890/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "indic-parler-tts", "input": "नमस्ते, आप कैसे हैं?", "voice": "divya"}' \
  --output speech.wav
```

### List Models
```bash
curl http://localhost:8890/v1/models
```

### List Voices
```bash
curl http://localhost:8890/v1/voices
```

## Available Voices

| Voice | Gender | Style |
|-------|--------|-------|
| divya | Female | Clear, expressive |
| leela | Female | High-pitched, cheerful |
| maya | Female | Calm, soothing |
| sita | Female | Professional, neutral |
| priya | Female | Warm, friendly |
| rohit | Male | Deep, authoritative |
| karan | Male | Warm, friendly |
| arjun | Male | Professional, clear |
| vijay | Male | Expressive, animated |

## Integration with LiveKit Agent

```python
from livekit.plugins import openai

indic_tts = openai.TTS(
    base_url="http://localhost:8890/v1",
    model="indic-parler-tts",
    voice="divya",
    api_key="indic-parler",
)
```
