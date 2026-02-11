"""
Minimal FastAPI Server for Indic TTS
"""

import io
import logging
from typing import Dict, Optional

import numpy as np
import scipy.io.wavfile
import torch
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import Response
from pydantic import BaseModel

from inference import IndicTTS, SUPPORTED_LANGUAGES, SPEAKERS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("IndicTTS-Server")

app = FastAPI(title="Indic TTS API", version="1.0.0")

# Global cache for loaded models: lang_code -> IndicTTS instance
# We load models on demand to save VRAM
model_cache: Dict[str, IndicTTS] = {}

class SynthesisRequest(BaseModel):
    text: str
    lang: str = "hi"
    speaker: str = "female"
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "नमस्ते, आप कैसे हैं?",
                "lang": "hi",
                "speaker": "female"
            }
        }

def get_model(lang: str) -> IndicTTS:
    """Get or load model for the specified language."""
    if lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Language '{lang}' not supported. Supported: {list(SUPPORTED_LANGUAGES.keys())}"
        )
        
    if lang not in model_cache:
        logger.info(f"Loading model for language: {lang}")
        try:
            # Initialize with default speaker 'female' - we can override during synthesis
            model_cache[lang] = IndicTTS(
                model_dir="./models",
                lang=lang,
                speaker="female",
                use_cuda=True
            )
        except Exception as e:
            logger.error(f"Failed to load model for {lang}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load model: {str(e)}"
            )
            
    return model_cache[lang]

@app.get("/health")
async def health_check():
    return {"status": "ok", "loaded_models": list(model_cache.keys())}

@app.get("/languages")
async def get_languages():
    return {
        "languages": SUPPORTED_LANGUAGES,
        "speakers": SPEAKERS
    }

@app.post("/synthesize")
async def synthesize(request: SynthesisRequest):
    """
    Synthesize text to speech
    Returns WAV audio bytes
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
        
    if request.speaker not in SPEAKERS:
         raise HTTPException(
            status_code=400, 
            detail=f"Speaker '{request.speaker}' not supported. Supported: {SPEAKERS}"
        )

    # Get model (load if not cached)
    tts_model = get_model(request.lang)
    
    try:
        logger.info(f"Synthesizing for lang={request.lang}, speaker={request.speaker}")
        # Generate audio
        # wav is numpy array, sr is int
        wav, sample_rate = tts_model.synthesize(
            text=request.text,
            speaker=request.speaker
        )
        
        # Convert numpy array to WAV bytes
        # We need to normalize if float, typically Coqui output is float32
        buffer = io.BytesIO()
        scipy.io.wavfile.write(buffer, sample_rate, wav)
        
        return Response(
            content=buffer.getvalue(),
            media_type="audio/wav"
        )
        
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8027)
