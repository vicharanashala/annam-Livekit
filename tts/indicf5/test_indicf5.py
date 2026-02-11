import torch
import numpy as np
import soundfile as sf
from transformers import AutoModel

# Load TTS model
repo_id = "ai4bharat/IndicF5"
print(f"Loading model from {repo_id}...")
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)

# Move to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
model = model.to(device)

# Reference audio path
ref_audio_path = "IndicF5/prompts/PAN_F_HAPPY_00001.wav"
ref_text = "ਭਹੰਪੀ ਵਿੱਚ ਸਮਾਰਕਾਂ ਦੇ ਭਵਨ ਨਿਰਮਾਣ ਕਲਾ ਦੇ ਵੇਰਵੇ ਗੁੰਝਲਦਾਰ ਅਤੇ ਹੈਰਾਨ ਕਰਨ ਵਾਲੇ ਹਨ, ਜੋ ਮੈਨੂੰ ਖੁਸ਼ ਕਰਦੇ  ਹਨ।"

# Text to synthesize
text_to_speak = "नमस्ते! संगीत की तरह जीवन भी खूबसूरत होता है, बस इसे सही ताल में जीना आना चाहिए."

print(f"Reference audio: {ref_audio_path}")
print(f"Reference text: {ref_text}")
print(f"Text to synthesize: {text_to_speak}")
print("Generating audio...")

# Generate speech
audio = model(text_to_speak, ref_audio_path=ref_audio_path, ref_text=ref_text)

# Normalize output and save
if audio.dtype == np.int16:
    audio = audio.astype(np.float32) / 32768.0

sf.write("namaste.wav", np.array(audio, dtype=np.float32), samplerate=24000)
print("Audio saved successfully: namaste.wav")
