#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ai4bharat/indic-seamless 
# ...........................................
import torch
import torchaudio
import soundfile as sf
from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText

# -------------------------
# Settings
# -------------------------
MODEL_ID = "ai4bharat/indic-seamless"
AUDIO_PATH = "sound9_female.wav"
OUTPUT_TXT = "output.txt"
TARGET_LANG = "mal"   # Malayalam

# Choose decoding mode:
# "deterministic"  -> Option 1
# "beam"           -> Option 2
DECODE_MODE = "beam"

# -------------------------
# Device
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print("Using device:", device)

# -------------------------
# Load model + processor
# -------------------------
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = SeamlessM4Tv2ForSpeechToText.from_pretrained(MODEL_ID).to(device)

# -------------------------
# Load audio
# -------------------------
audio, sample_rate = sf.read(AUDIO_PATH)

# convert to torch tensor
waveform = torch.tensor(audio)

# if stereo → mono
if len(waveform.shape) > 1:
    waveform = waveform.mean(dim=1)

# resample if needed
if sample_rate != 16000:
    waveform = torch.nn.functional.interpolate(
        waveform.unsqueeze(0).unsqueeze(0),
        size=int(len(waveform) * 16000 / sample_rate),
        mode="linear",
        align_corners=False
    ).squeeze()

waveform = waveform.float()

# -------------------------
# Preprocess
# -------------------------
inputs = processor(
    audio=waveform.numpy(),
    sampling_rate=16000,
    return_tensors="pt"
).to(device)

# -------------------------
# Generate transcription
# -------------------------
with torch.no_grad():

    if DECODE_MODE == "deterministic":
        # ✅ OPTION 1: Deterministic decoding (more acoustic, less smoothing)
        generated_tokens = model.generate(
            **inputs,
            tgt_lang=TARGET_LANG,
            temperature=0.0,
            do_sample=False
        )

    elif DECODE_MODE == "beam":
        # ✅ OPTION 2: Beam search (more accurate than greedy)
        generated_tokens = model.generate(
            **inputs,
            tgt_lang=TARGET_LANG,
            num_beams=5,
            early_stopping=True
        )

    else:
        # Default (original behaviour)
        generated_tokens = model.generate(
            **inputs,
            tgt_lang=TARGET_LANG
        )

transcription = processor.batch_decode(
    generated_tokens,
    skip_special_tokens=True
)[0]

# -------------------------
# Print + Save to TXT
# -------------------------
print("\nTranscription:\n", transcription)

with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
    f.write(transcription)

print("\nSaved to:", OUTPUT_TXT)

# ................................
# ==========================================================
# 🔥 LATENCY MEASUREMENT BLOCK (Add at End of Script)
# ==========================================================
import time

print("\n----- Latency Analysis -----")

# Calculate audio duration
audio_duration = waveform.shape[0] / 16000
print(f"Audio Duration: {audio_duration:.2f} seconds")

# Synchronize GPU before timing (important for CUDA)
if device == "cuda":
    torch.cuda.synchronize()

start_time = time.time()

with torch.no_grad():

    if DECODE_MODE == "deterministic":
        generated_tokens = model.generate(
            **inputs,
            tgt_lang=TARGET_LANG,
            temperature=0.0,
            do_sample=False
        )

    elif DECODE_MODE == "beam":
        generated_tokens = model.generate(
            **inputs,
            tgt_lang=TARGET_LANG,
            num_beams=8,
            early_stopping=True
        )

    else:
        generated_tokens = model.generate(
            **inputs,
            tgt_lang=TARGET_LANG
        )

# Synchronize GPU after inference
if device == "cuda":
    torch.cuda.synchronize()

end_time = time.time()

inference_time = end_time - start_time
rtf = inference_time / audio_duration

print(f"Inference Time: {inference_time:.3f} seconds")
print(f"Real-Time Factor (RTF): {rtf:.3f}")

if rtf < 1:
    print("Model runs faster than real-time ✅")
else:
    print("Model runs slower than real-time ⚠️")

