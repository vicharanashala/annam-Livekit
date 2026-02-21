"""
OpenAI-API-style FastAPI backend for TTS
- Output: ONLY raw PCM (s16le) @ 24kHz (audio/pcm)
- Streaming: ALWAYS streams PCM chunks
- Text chunking: first chunk is *tiny* (3–4 words) to get audio out fast,
  remaining chunks are larger for throughput.
- Speed optimizations:
  - model loaded once on startup
  - torch.inference_mode()
  - TF32 (on Ampere+)
  - torch.compile (optional, guarded)
  - warmup on startup (optional, guarded)
  - fixed nfe_step low (fast)
  - optional autocast (if your model supports it)
  - avoids re-creating resampler repeatedly
  - minimal overhead in generator

Endpoint:
POST /v1/audio/speech
Body:
{
  "model": "ai4bharat/IndicF5",
  "input": "text ...",
  "voice": "default",
  "speed": 1.0,
  "ref_audio_path": null,
  "ref_text": null
}

Response:
StreamingResponse(media_type="audio/pcm")
Headers include X-Sample-Rate: 24000

Run:
  uvicorn main:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import time
import re
import requests
from typing import Optional, Generator, List

import numpy as np
import torch
import torchaudio
import soundfile as sf

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


# =============================================================================
# 1) PATH + IndicF5 Imports
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDIC_F5_PATH = os.path.join(BASE_DIR, "IndicF5")

if INDIC_F5_PATH not in sys.path:
    sys.path.append(INDIC_F5_PATH)

try:
    from transformers import AutoModel
    from f5_tts.infer.utils_infer import preprocess_ref_audio_text, chunk_text
    from f5_tts.model.utils import convert_char_to_pinyin
except Exception as e:
    raise RuntimeError(
        "Could not import IndicF5 / f5_tts modules. Ensure IndicF5 repo is present "
        f"at: {INDIC_F5_PATH} and dependencies installed.\nError: {e}"
    )

app = FastAPI(title="OpenAI-style PCM TTS API")

device = "cuda" if torch.cuda.is_available() else "cpu"

REPO_ID = "ai4bharat/IndicF5"
DEFAULT_REF_AUDIO_PATH = os.path.join(BASE_DIR, "IndicF5", "prompts", "PAN_F_HAPPY_00001.wav")
DEFAULT_REF_TEXT = "ਭਹੰਪੀ ਵਿੱਚ ਸਮਾਰਕਾਂ ਦੇ ਭਵਨ ਨਿਰਮਾਣ ਕਲਾ ਦੇ ਵੇਰਵੇ ਗੁੰਝਲਦਾਰ ਅਤੇ ਹੈਰਾਨ ਕਰਨ ਵਾਲੇ ਹਨ, ਜੋ ਮੈਨੂੰ ਖੁਸ਼ ਕਰਦੇ ਹਨ।"

TARGET_SAMPLE_RATE = 24000
HOP_LENGTH = 256
TARGET_RMS = 0.1

# --- speed knobs (tune) ---
DEFAULT_NFE_STEP = 16          # lower = faster, more artifacts risk
CFG_STRENGTH = 2.0
SWAY_SAMPLING_COEF = -1.0

# --- first-chunk policy ---
FIRST_CHUNK_WORDS = 4         # 3–4 words requested; set 4
FIRST_CHUNK_MIN_CHARS = 12    # avoid too tiny if language has short tokens

# Global model
model = None

# Cache resamplers to avoid re-creation overhead
_resampler_cache = {}


# =============================================================================
# 2) Request schema (OpenAI-ish)
# =============================================================================
class SpeechRequest(BaseModel):
    model: str = Field(default=REPO_ID)
    input: str = Field(..., min_length=1)
    voice: str = Field(default="default")
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    ref_audio_path: Optional[str] = None
    ref_text: Optional[str] = None


# =============================================================================
# 3) Torch / CUDA optimizations
# =============================================================================
def _enable_torch_optimizations():
    # Determinism off for speed
    torch.backends.cudnn.benchmark = True

    if torch.cuda.is_available():
        # TF32 can significantly speed up matmul/conv on Ampere+
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Optional: reduce overhead in some workloads
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def _maybe_compile(m):
    """
    torch.compile can speed up steady-state; may increase startup time and memory.
    Keep guarded; if it fails, we continue without compile.
    """
    if not hasattr(torch, "compile"):
        return m
    try:
        # mode choices: "reduce-overhead" often good for inference servers
        return torch.compile(m, mode="reduce-overhead", fullgraph=False)
    except Exception:
        return m


# =============================================================================
# 4) Startup load + warmup
# =============================================================================
@app.on_event("startup")
async def startup_load():
    global model
    _enable_torch_optimizations()

    print(f"[startup] Loading {REPO_ID} on {device} ...")
    model = AutoModel.from_pretrained(REPO_ID, trust_remote_code=True).to(device)
    model.eval()

    # Optional compile (comment out if you prefer predictable startup)
    model = _maybe_compile(model)

    # Optional warmup: tiny run to trigger kernels/allocations
    try:
        _warmup()
        print("[startup] Warmup done.")
    except Exception as e:
        print(f"[startup] Warmup skipped/failed: {e}")

    print("[startup] Model ready.")


def _warmup():
    # A very small warmup to reduce first-request latency.
    # Uses default ref prompt; if missing file, warmup fails (safe).
    if not os.path.exists(DEFAULT_REF_AUDIO_PATH):
        return

    _ = list(
        _pcm_stream_generator(
            text="Hello world.",
            speed=1.0,
            ref_audio_path=DEFAULT_REF_AUDIO_PATH,
            ref_text=DEFAULT_REF_TEXT,
            nfe_step=DEFAULT_NFE_STEP,
            max_chunks=1,  # only first chunk
        )
    )


# =============================================================================
# 5) Number Transliteration (Hindi)
# =============================================================================
NUMBERS_0_TO_99 = [
    "शून्य", "एक", "दो", "तीन", "चार", "पांच", "छह", "सात", "आठ", "नौ",
    "दस", "ग्यारह", "बारह", "तेरह", "चौदह", "पंद्रह", "सोलह",
    "सत्रह", "अठारह", "उन्नीस",
    "बीस", "इक्कीस", "बाईस", "तेइस", "चौबीस", "पच्चीस",
    "छब्बीस", "सत्ताईस", "अट्ठाईस", "उनतीस",
    "तीस", "इकतीस", "बत्तीस", "तैंतीस", "चौंतीस", "पैंतीस",
    "छत्तीस", "सैंतीस", "अड़तीस", "उनतालीस",
    "चालीस", "इकतालीस", "बयालीस", "तैंतालीस", "चवालीस", "पैंतालीस",
    "छियालीस", "सैंतालीस", "अड़तालीस", "उनचास",
    "पचास", "इक्यावन", "बावन", "तिरेपन", "चौवन", "पचपन",
    "छप्पन", "सत्तावन", "अट्ठावन", "उनसठ",
    "साठ", "इकसठ", "बासठ", "तिरसठ", "चौंसठ", "पैंसठ",
    "छियासठ", "सड़सठ", "अड़सठ", "उनहत्तर",
    "सत्तर", "इकहत्तर", "बहत्तर", "तिहत्तर", "चौहत्तर", "पचहत्तर",
    "छिहत्तर", "सतहत्तर", "अठहत्तर", "उन्यासी",
    "अस्सी", "इक्यासी", "बयासी", "तिरासी", "चौरासी", "पचासी",
    "छियासी", "सत्तासी", "अट्ठासी", "नवासी",
    "नब्बे", "इक्यानवे", "बानवे", "तिरानवे", "चौरानवे",
    "पचानवे", "छियानवे", "सत्तानवे", "अट्ठानवे", "निन्यानवे"
]

def two_digit_word(n):
    return NUMBERS_0_TO_99[n]

def three_digit_word(n):
    hundred = n // 100
    rest = n % 100
    result = ""
    if hundred:
        result += NUMBERS_0_TO_99[hundred] + " सौ"
    if rest:
        result += (" " if result else "") + two_digit_word(rest)
    return result

def number_to_words_indian(n):
    # Handle negatives
    if n < 0:
        return "ऋण " + number_to_words_indian(abs(n))
    
    # Handle zero explicitly
    if n == 0:
        return "शून्य"
    
    # Handle floats
    n = int(n)

    crore = n // 10000000
    n %= 10000000
    lakh = n // 100000
    n %= 100000
    thousand = n // 1000
    n %= 1000
    remainder = n

    result = ""

    if crore:
        # FIX: Recursive call allows for numbers > 99 Crores (e.g., 100 Crore)
        crore_text = number_to_words_indian(crore)
        if crore_text == "शून्य": crore_text = "" 
        result += crore_text + " करोड़ "

    if lakh:
        result += two_digit_word(lakh) + " लाख "
    
    if thousand:
        result += two_digit_word(thousand) + " हजार "
    
    if remainder:
        result += three_digit_word(remainder)

    return result.strip()

def convert_number(number):
    return number_to_words_indian(number)


def convert_number_with_decimal(match):
    text = match.group()
    if '.' in text:
        try:
            parts = text.split('.')
            whole = int(parts[0])
            frac = parts[1]
            # Whole part -> words
            words = number_to_words_indian(whole) + " दशमलव"
            # Fractional part -> read digits
            for digit in frac:
                words += " " + NUMBERS_0_TO_99[int(digit)]
            return words
        except Exception:
            return text
    else:
        return number_to_words_indian(int(text))


def normalize_text(text: str) -> str:
    """
    Applies rule-based normalization to convert symbols/patterns into spoken Hindi (Devanagari).
    Order of replacements is critical (longest matches first).
    """
    
    # 1. Acronyms: All Caps (2 or more letters) -> Add dots
    # e.g. FAQ -> F.A.Q., PH -> P.H.
    def add_dots_to_acronyms(match):
        word = match.group(0)
        return ".".join(list(word)) + "."
    
    # Pattern: discrete words, all caps, length >= 2
    text = re.sub(r'\b[A-Z]{2,}\b', add_dots_to_acronyms, text)
    
    # 2. Short consonant words (1-3 letters, no vowels aeiou/y) -> Add dots
    # e.g. ml -> m.l.
    def add_dots_to_short_consonants(match):
        word = match.group(0)
        return ".".join(list(word)) + " "
    
    # Regex: word boundary, 1-3 lowercase consonants (no aeiouy)
    # consonants: bcdfghjklmnpqrstvwxz
    consonants = "bcdfghjklmnpqrstvwxz"
    pattern = r'\b[' + consonants + r']{1,3}\b'
    text = re.sub(pattern, add_dots_to_short_consonants, text)



    
    replacements = [
        # Units / Compound Symbols
        ("°C", " डिग्री सेल्सियस"),
        ("°", " डिग्री"),
        ("%", " प्रतिशत"),

        # Formatting / Punctuation
        ("\n", ". "),  # Newline -> Dot Space
        ("।", "."),    # Danda -> Dot
        ("*", ""),     # Remove asterisks
        ("#", " "),    # Hashtag -> Space
        ("-", " ")     # Hyphen -> Space
    ]

    for old, new in replacements:
        text = text.replace(old, new)

    # Specific Slash Handling: replace '/' with ' प्रति ' ONLY if between non-digits (e.g., words/units)
    # Ignored: 24/7, 1/2 (fractions/dates)
    # Matched: मीटर/सेकंड -> मीटर प्रति सेकंड

    text = re.sub(
    r'(\d+(?:\.\d+)?\s[\u0900-\u097F]+)\/([\u0900-\u097F]+)',
    r'\1 प्रति \2',
    text
)
    pattern_hms = r'\b(\d{2}):(\d{2}):(\d{2})\b'
    text = re.sub(pattern_hms, r'\1 बज के \2 मिनट \3 सेकंड', text)
    pattern = r'\b(\d{2}):(\d{2})\b'
    text = re.sub(pattern, r'\1 बज के \2 मिनट', text)
    # Number Transliteration (Decimal aware)
    text = re.sub(r'\d+(\.\d+)?', convert_number_with_decimal, text)
   

    # 3. Transliterate English to Indic (Hindi) via API
    try:
        url = "https://mesne-unlicentiously-allie.ngrok-free.dev/xlit/transliterate/en-to-indic"
        payload = {
            "text": text,
            "target_languages": ["hi"],
            "beam_width": 4
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, json=payload, headers=headers, timeout=3)
        if response.status_code == 200:
             data = response.json()
             # expected format: {"success":true,"results":{"hi":"..."}, ...}
             if data.get("success") and "hi" in data.get("results", {}):
                 transliterated = data["results"]["hi"]
                 print(f"[TTS] Transliteration success: '{text}' -> '{transliterated}'")
                 text = transliterated
    except Exception as e:
        print(f"[TTS] Transliteration API failed: {e}. Falling back to original text.")

    # Cleanup multi-spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text



# =============================================================================
# 6) Audio utils
# =============================================================================
def _get_resampler(sr_from: int, sr_to: int):
    key = (sr_from, sr_to, device)
    if key in _resampler_cache:
        return _resampler_cache[key]
    r = torchaudio.transforms.Resample(sr_from, sr_to)
    _resampler_cache[key] = r
    return r


def _float_to_int16_pcm_bytes(wave: np.ndarray) -> bytes:
    wave = np.clip(wave, -1.0, 1.0)
    return (wave * 32767.0).astype(np.int16).tobytes()


# =============================================================================
# 7) Core generator (streams PCM bytes)
# =============================================================================
def _pcm_stream_generator(
    text: str,
    speed: float,
    ref_audio_path: str,
    ref_text: str,
    nfe_step: int,
    max_chunks: Optional[int] = None,
) -> Generator[bytes, None, None]:
    """
    Streams int16 PCM chunks (s16le) at 24kHz.
    aligned with wav_server.py logic
    """
    # Normalize text (Symbols -> Hindi, Numbers -> Hindi words)
    text = normalize_text(text)
    print(f"[TTS] Normalized text: {text}")

    # 1) preprocess reference
    ref_audio_fname, ref_text_final = preprocess_ref_audio_text(
        ref_audio_path, ref_text, show_info=lambda _: None
    )

    audio_np, sr = sf.read(ref_audio_fname)
    if audio_np.ndim == 1:
        audio = torch.from_numpy(audio_np).float().unsqueeze(0)  # [1, T]
    else:
        audio = torch.from_numpy(audio_np).float().t()  # [C, T]

    # Parameters from wav_server.py
    # target_sample_rate = 24000 (global TARGET_SAMPLE_RATE)
    # hop_length = 256 (global HOP_LENGTH)
    # target_rms = 0.1 (global TARGET_RMS)
    # speed already passed in

    # wav_server.py logic for max_chars (Before resampling)
    # audio.shape[-1] and sr correspond to the file read
    sec = max(audio.shape[-1] / float(sr), 1e-6)
    max_chars = int(len(ref_text_final.encode("utf-8")) / sec * (25 - sec))
    # max_chars = max(40, max_chars) # wav_server doesn't enforce min 40, but it shouldn't hurt. 
    # Let's stick to wav_server logic strictly where possible.

    # 2) Chunk text - use standard chunk_text
    gen_text_batches = chunk_text(text, max_chars=max_chars)
    print(f"[TTS] Chunked into {len(gen_text_batches)} parts.")

    if not gen_text_batches:
        return

    # 3) Process Audio (Resampling and RMS)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < TARGET_RMS:
        audio = audio * TARGET_RMS / rms

    if sr != TARGET_SAMPLE_RATE:
        # Use simpler torchaudio resample as in wav_server.py
        # check if cache is okay, or just instantiate
        resampler = torchaudio.transforms.Resample(sr, TARGET_SAMPLE_RATE)
        audio = resampler(audio)
    
    audio = audio.to(device)

    ref_audio_len = audio.shape[-1] // HOP_LENGTH
    ref_text_len = max(1, len(ref_text_final.encode("utf-8")))

    first_sent = False
    for i, gen_text in enumerate(gen_text_batches):
        if max_chunks is not None and i >= max_chunks:
            break

        t0 = time.time()

        # Prepare text
        text_list = [ref_text_final + gen_text]
        final_text_list = convert_char_to_pinyin(text_list)

        gen_text_len = len(gen_text.encode("utf-8"))
        duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / speed)

        with torch.inference_mode():
            generated, _ = model.ema_model.sample(
                cond=audio,
                text=final_text_list,
                duration=duration,
                steps=nfe_step,
                cfg_strength=CFG_STRENGTH,
                sway_sampling_coef=SWAY_SAMPLING_COEF,
            )
            
            generated = generated.to(torch.float32)
            generated = generated[:, ref_audio_len:, :]
            mel = generated.permute(0, 2, 1)
            wave = model.vocoder.decode(mel)

            # wav_server.py has this checks. 
            if rms < TARGET_RMS:
                wave = wave * rms / TARGET_RMS

            wave = wave.squeeze().detach().cpu().numpy().astype(np.float32)
            pcm_bytes = _float_to_int16_pcm_bytes(wave)

        if not first_sent:
            print(f"[TTS] first chunk ready in {time.time() - t0:.3f}s ({len(pcm_bytes)} bytes)")
            first_sent = True
        else:
            print(f"[TTS] chunk {i+1}/{len(gen_text_batches)} in {time.time() - t0:.3f}s ({len(pcm_bytes)} bytes)")

        yield pcm_bytes


# =============================================================================
# 8) OpenAI-style endpoint (ONLY PCM)
# =============================================================================
@app.post("/v1/audio/speech")
async def audio_speech(req: SpeechRequest, request: Request):
    """
    Always returns streamed raw PCM (int16, 24kHz).
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    ref_audio_path = req.ref_audio_path or DEFAULT_REF_AUDIO_PATH
    ref_text = req.ref_text or DEFAULT_REF_TEXT

    if not os.path.exists(ref_audio_path):
        raise HTTPException(status_code=400, detail=f"ref_audio_path not found: {ref_audio_path}")

    # Lower steps for speed; you can also make it dynamic based on text length
    nfe_step = DEFAULT_NFE_STEP

    def gen():
        yield from _pcm_stream_generator(
            text=req.input,
            speed=req.speed,
            ref_audio_path=ref_audio_path,
            ref_text=ref_text,
            nfe_step=nfe_step,
        )

    return StreamingResponse(
        gen(),
        media_type="audio/pcm",
        headers={
            "X-Sample-Rate": str(TARGET_SAMPLE_RATE),
            "X-Audio-Format": "pcm_s16le",
        },
    )


# Optional: minimal compatibility endpoint
@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": REPO_ID, "object": "model", "owned_by": "local"}]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
