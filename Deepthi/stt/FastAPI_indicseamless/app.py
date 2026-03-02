#!/usr/bin/env python
# coding: utf-8

import torch
import soundfile as sf
import torchaudio
import io
from fastapi import FastAPI, UploadFile, File, Form
from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText

# -------------------------
# Config
# -------------------------
MODEL_ID = "ai4bharat/indic-seamless"
TARGET_SAMPLE_RATE = 16000

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -------------------------
# Load model ONCE at startup
# -------------------------
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = SeamlessM4Tv2ForSpeechToText.from_pretrained(MODEL_ID).to(device)
model.eval()

# -------------------------
# FastAPI App
# -------------------------
app = FastAPI(title="Indic Seamless STT API")


# -------------------------
# Core Transcription Logic
# -------------------------
def transcribe_audio_logic(audio_bytes, target_lang, decode_mode):

    audio, sample_rate = sf.read(io.BytesIO(audio_bytes))

    # Force float32 (IMPORTANT FIX)
    audio = audio.astype("float32")

    waveform = torch.from_numpy(audio)

    # Stereo → Mono
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=1)

    # Resample if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=16000
        )
        waveform = resampler(waveform)

    waveform = waveform.float()

    inputs = processor(
        audio=waveform.numpy(),
        sampling_rate=16000,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():

        if decode_mode == "deterministic":
            generated_tokens = model.generate(
                **inputs,
                tgt_lang=target_lang,
                temperature=0.0,
                do_sample=False
            )

        elif decode_mode == "beam":
            generated_tokens = model.generate(
                **inputs,
                tgt_lang=target_lang,
                num_beams=5,
                early_stopping=True
            )

        else:
            generated_tokens = model.generate(
                **inputs,
                tgt_lang=target_lang
            )

    transcription = processor.batch_decode(
        generated_tokens,
        skip_special_tokens=True
    )[0]

    return transcription

# -------------------------
# API Endpoint
# -------------------------
@app.post("/transcribe/")
async def transcribe_audio(
    file: UploadFile = File(...),
    target_lang: str = Form("hin"),
    decode_mode: str = Form("beam")
):
    try:
        audio_bytes = await file.read()

        transcription = transcribe_audio_logic(
            audio_bytes,
            target_lang,
            decode_mode
        )

        return {
            "language": target_lang,
            "transcription": transcription
        }

    except Exception as e:
        return {"error": str(e)}


# ...............................................................................................
# #!/usr/bin/env python
# # coding: utf-8

# # In[ ]:


# import torch
# import soundfile as sf
# from fastapi import FastAPI, UploadFile, File, Form
# from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText
# import io

# # -------------------------
# # Config
# # -------------------------
# MODEL_ID = "ai4bharat/indic-seamless"

# device = "cuda" if torch.cuda.is_available() else "cpu"
# print("Using device:", device)

# # -------------------------
# # Load model ONCE at startup
# # -------------------------
# processor = AutoProcessor.from_pretrained(MODEL_ID)
# model = SeamlessM4Tv2ForSpeechToText.from_pretrained(MODEL_ID).to(device)
# model.eval()

# app = FastAPI(title="Indic Seamless STT API")


# # -------------------------
# # API Endpoint
# # -------------------------
# @app.post("/transcribe/")
# async def transcribe_audio(
#     file: UploadFile = File(...),
#     target_lang: str = Form("hin"),   # default Hindi
#     decode_mode: str = Form("beam")   # beam / deterministic
# ):

#     # Read audio
#     audio_bytes = await file.read()
#     audio, sample_rate = sf.read(io.BytesIO(audio_bytes))

#     waveform = torch.tensor(audio)

#     # Stereo → mono
#     if len(waveform.shape) > 1:
#         waveform = waveform.mean(dim=1)

#     # Resample if needed
#     if sample_rate != 16000:
#         waveform = torch.nn.functional.interpolate(
#             waveform.unsqueeze(0).unsqueeze(0),
#             size=int(len(waveform) * 16000 / sample_rate),
#             mode="linear",
#             align_corners=False
#         ).squeeze()

#     waveform = waveform.float()

#     inputs = processor(
#         audio=waveform.numpy(),
#         sampling_rate=16000,
#         return_tensors="pt"
#     ).to(device)

#     with torch.no_grad():

#         if decode_mode == "deterministic":
#             generated_tokens = model.generate(
#                 **inputs,
#                 tgt_lang=target_lang,
#                 temperature=0.0,
#                 do_sample=False
#             )

#         elif decode_mode == "beam":
#             generated_tokens = model.generate(
#                 **inputs,
#                 tgt_lang=target_lang,
#                 num_beams=5,
#                 early_stopping=True
#             )

#         else:
#             generated_tokens = model.generate(
#                 **inputs,
#                 tgt_lang=target_lang
#             )

#     transcription = processor.batch_decode(
#         generated_tokens,
#         skip_special_tokens=True
#     )[0]

#     return {
#         "language": target_lang,
#         "transcription": transcription
#     }

