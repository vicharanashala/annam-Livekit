# Indic TTS

Minimal inference setup for [AI4Bharat Indic-TTS](https://github.com/AI4Bharat/Indic-TTS) - Text-to-Speech for 13 Indian languages.

## Supported Languages

| Code | Language  | Code | Language  |
|------|-----------|------|-----------|
| as   | Assamese  | ml   | Malayalam |
| bn   | Bengali   | mni  | Manipuri  |
| brx  | Bodo      | mr   | Marathi   |
| gu   | Gujarati  | or   | Odia      |
| hi   | Hindi     | raj  | Rajasthani|
| kn   | Kannada   | ta   | Tamil     |
|      |           | te   | Telugu    |

## Quick Setup

```bash
# 1. Setup environment
chmod +x setup_env.sh
./setup_env.sh

# 2. Activate environment
source venv/bin/activate

# 3. Download model weights (e.g., Hindi)
python download_models.py --lang hi

# 4. Run inference
python inference.py --text "नमस्ते, आप कैसे हैं?" --lang hi --output output.wav
```

## Usage

### Command Line

```bash
# Hindi
python inference.py --text "नमस्ते" --lang hi --speaker female --output hindi.wav

# Tamil with male voice
python inference.py --text "வணக்கம்" --lang ta --speaker male --output tamil.wav

# Telugu
python inference.py --text "నమస్కారం" --lang te --speaker female --output telugu.wav
```

### Python API

```python
from inference import IndicTTS

# Initialize with default female speaker
tts = IndicTTS(model_dir="./models", lang="hi", speaker="female")

# Synthesize
audio, sample_rate = tts.synthesize(
    text="नमस्ते, आप कैसे हैं?",
    output_path="output.wav"
)

# Use male speaker
audio, sample_rate = tts.synthesize(
    text="नमस्ते",
    output_path="output_male.wav",
    speaker="male"
)
```

## API Server

Run the FastAPI server:

```bash
# Start server
python server.py
```

### Endpoints

- `GET /health`: Check status
- `GET /languages`: List supported languages
- `POST /synthesize`: Generate audio

**Example Request:**
```bash
curl -X POST "http://localhost:8000/synthesize" \
     -H "Content-Type: application/json" \
     -d '{"text": "नमस्ते", "lang": "hi", "speaker": "female"}' \
     --output output.wav
```

## Model Architecture

- **Acoustic Model**: FastPitch (parallel TTS with pitch prediction)
- **Vocoder**: HiFi-GAN V1 (high-fidelity neural vocoder)

## References

- [AI4Bharat Indic-TTS Paper](https://arxiv.org/abs/2211.09536) (ICASSP 2023)
- [Coqui TTS](https://github.com/coqui-ai/TTS)
