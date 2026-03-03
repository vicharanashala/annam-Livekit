#!/usr/bin/env python
# coding: utf-8

import torch
import time
import soundfile as sf
import torchaudio
from transformers import AutoModel

# -------------------------
# Settings
# -------------------------
MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
AUDIO_PATH = "sound1_female.wav"
TARGET_LANG = "ml"
TARGET_SAMPLE_RATE = 16000

# -------------------------
# Device
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -------------------------
# Load Model
# -------------------------
model = AutoModel.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
).to(device)

model.eval()

# -------------------------
# Load Audio (NO TORCHCODEC)
# -------------------------
wav, sr = sf.read(AUDIO_PATH)

# Convert to torch tensor
wav = torch.tensor(wav).float()

# If stereo → convert to mono
if wav.ndim == 2:
    wav = wav.mean(dim=1)

# Add batch dimension → (1, T)
wav = wav.unsqueeze(0)

# Resample if needed
if sr != TARGET_SAMPLE_RATE:
    wav = torchaudio.functional.resample(
        wav, sr, TARGET_SAMPLE_RATE
    )
    sr = TARGET_SAMPLE_RATE

wav = wav.to(device)

# Calculate audio duration
audio_duration = wav.shape[1] / TARGET_SAMPLE_RATE
print(f"\nAudio Duration: {audio_duration:.2f} seconds")

# ==========================================================
# 🔥 CTC LATENCY
# ==========================================================
if device == "cuda":
    torch.cuda.synchronize()

start_time = time.time()

with torch.no_grad():
    transcription_ctc = model(wav, TARGET_LANG, "ctc")

if device == "cuda":
    torch.cuda.synchronize()

end_time = time.time()

ctc_inference_time = end_time - start_time
ctc_rtf = ctc_inference_time / audio_duration

print("\n========== CTC ==========")
print("CTC Transcription:", transcription_ctc)
print(f"CTC Inference Time: {ctc_inference_time:.3f} sec")
print(f"CTC RTF: {ctc_rtf:.3f}")

# ==========================================================
# 🔥 RNNT LATENCY
# ==========================================================
if device == "cuda":
    torch.cuda.synchronize()

start_time = time.time()

with torch.no_grad():
    transcription_rnnt = model(wav, TARGET_LANG, "rnnt")

if device == "cuda":
    torch.cuda.synchronize()

end_time = time.time()

rnnt_inference_time = end_time - start_time
rnnt_rtf = rnnt_inference_time / audio_duration

print("\n========== RNNT ==========")
print("RNNT Transcription:", transcription_rnnt)
print(f"RNNT Inference Time: {rnnt_inference_time:.3f} sec")
print(f"RNNT RTF: {rnnt_rtf:.3f}")

# -------------------------
# Save Output
# -------------------------
with open("final_transcription.txt", "w", encoding="utf-8") as f:
    f.write("CTC Transcription:\n")
    f.write(str(transcription_ctc) + "\n\n")
    f.write("RNNT Transcription:\n")
    f.write(str(transcription_rnnt) + "\n")

print("\nSaved to final_transcription.txt")

# -------------------------
# Speed Comparison
# -------------------------
print("\n========== COMPARISON ==========")
if ctc_rtf < rnnt_rtf:
    print("CTC is faster than RNNT")
else:
    print("RNNT is faster than CTC")