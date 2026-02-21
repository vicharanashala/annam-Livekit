import os
import sys
import time
import io
import shutil
import tempfile
import torch
import torchaudio
import soundfile as sf
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse, HTMLResponse, Response
from pydantic import BaseModel
from typing import Optional
from transformers import AutoModel

# -----------------------------------------------------------------------------
# 1. PATH SETUP (Adjust based on your folder structure)
# -----------------------------------------------------------------------------
# We assume the directory structure:
# /project
#    /app/main.py
#    /IndicF5/ (The cloned repo for the model logic)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDIC_F5_PATH = os.path.join(BASE_DIR, "IndicF5")

if INDIC_F5_PATH not in sys.path:
    sys.path.append(INDIC_F5_PATH)

try:
    # Attempt to import necessary modules from the F5-TTS library structure
    from f5_tts.infer.utils_infer import (
        preprocess_ref_audio_text,
        chunk_text,
    )
    from f5_tts.model.utils import convert_char_to_pinyin
except ImportError as e:
    print("CRITICAL ERROR: Could not import f5_tts modules.")
    print(f"Ensure 'IndicF5' or 'f5-tts' is in your python path. Path used: {INDIC_F5_PATH}")
    raise e

# -----------------------------------------------------------------------------
# 2. MODEL LOADER
# -----------------------------------------------------------------------------
app = FastAPI(title="IndicF5 TTS API")

# Global variables for model
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# Configuration
REPO_ID = "ai4bharat/IndicF5"
REF_AUDIO_PATH = os.path.join(BASE_DIR, "IndicF5/prompts/PAN_F_HAPPY_00001.wav")
REF_TEXT = "ਭਹੰਪੀ ਵਿੱਚ ਸਮਾਰਕਾਂ ਦੇ ਭਵਨ ਨਿਰਮਾਣ ਕਲਾ ਦੇ ਵੇਰਵੇ ਗੁੰਝਲਦਾਰ ਅਤੇ ਹੈਰਾਨ ਕਰਨ ਵਾਲੇ ਹਨ, ਜੋ ਮੈਨੂੰ ਖੁਸ਼ ਕਰਦੇ ਹਨ।"

@app.on_event("startup")
async def load_models():
    global model
    print(f"Loading IndicF5 on {device}...")
    
    try:
        # Load model using transformers AutoModel (same as main.py)
        model = AutoModel.from_pretrained(REPO_ID, trust_remote_code=True)
        model = model.to(device)
        print("Model loaded successfully.")
        
    except Exception as e:
        print(f"Failed to load model: {e}")
        # Print full traceback to help debug further issues
        import traceback
        traceback.print_exc()

# -----------------------------------------------------------------------------
# 3. API ENDPOINTS
# -----------------------------------------------------------------------------
class TTSRequest(BaseModel):
    text: str
    speed: float = 1.0
    ref_audio_path: Optional[str] = None
    ref_text: Optional[str] = None

@app.post("/tts")
async def tts_endpoint(req: TTSRequest):
    """
    Streaming Endpoint.
    Returns raw PCM (Int16, 24kHz) to be played immediately by the browser.
    """
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Generator for StreamingResponse
    def audio_generator():
        try:
            # 1. Prepare Reference Audio
            ref_audio_fname, ref_text_final = preprocess_ref_audio_text(
                REF_AUDIO_PATH, REF_TEXT, show_info=lambda x: None
            )
            
            # Load Audio Tensor using soundfile to avoid TorchCodec errors
            audio_np, sr = sf.read(ref_audio_fname)
            if len(audio_np.shape) == 1:
                # Mono [T] -> [1, T]
                audio = torch.from_numpy(audio_np).float().unsqueeze(0)
            else:
                # Stereo [T, C] -> [C, T]
                audio = torch.from_numpy(audio_np).float().t()
            
            # Parameters
            target_sample_rate = 24000
            hop_length = 256
            target_rms = 0.1
            speed = req.speed if hasattr(req, 'speed') else model.config.speed
            nfe_step = 8  # Optimized for speed
            
            # Max chars calculation
            max_chars = int(len(ref_text_final.encode("utf-8")) / (audio.shape[-1] / sr) * (25 - audio.shape[-1] / sr))
            
            # 2. Chunk the Input Text
            gen_text_batches = chunk_text(req.text, max_chars=max_chars)
            print(f"[TTS] Chunked into {len(gen_text_batches)} parts.")
            
            # 3. Process Audio
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            
            rms = torch.sqrt(torch.mean(torch.square(audio)))
            if rms < target_rms:
                audio = audio * target_rms / rms
            if sr != target_sample_rate:
                resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
                audio = resampler(audio)
            
            audio = audio.to(device)  # Cond Audio
            
            # Iterate chunks
            first_chunk = True
            for i, gen_text in enumerate(gen_text_batches):
                chunk_start = time.time()
                
                # Prepare text
                text_list = [ref_text_final + gen_text]
                final_text_list = convert_char_to_pinyin(text_list)
                
                # Calculate duration
                ref_audio_len = audio.shape[-1] // hop_length
                ref_text_len = len(ref_text_final.encode("utf-8"))
                gen_text_len = len(gen_text.encode("utf-8"))
                duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / speed)
                
                # Inference
                with torch.inference_mode():
                    generated, _ = model.ema_model.sample(
                        cond=audio,
                        text=final_text_list,
                        duration=duration,
                        steps=nfe_step,
                        cfg_strength=2.0,
                        sway_sampling_coef=-1.0,
                    )
                    
                    generated = generated.to(torch.float32)
                    generated = generated[:, ref_audio_len:, :]
                    generated_mel_spec = generated.permute(0, 2, 1)
                    generated_wave = model.vocoder.decode(generated_mel_spec)
                    
                    if rms < target_rms:
                        generated_wave = generated_wave * rms / target_rms
                    
                    generated_wave = generated_wave.squeeze().cpu().numpy()
                    
                    # Convert to Int16 PCM
                    audio_int16 = (generated_wave * 32767).astype(np.int16)
                    yield audio_int16.tobytes()
                    
                    if first_chunk:
                        print(f"[TTS] First chunk ready in {time.time() - chunk_start:.3f}s")
                        first_chunk = False
                    print(f"[TTS] Chunk {i+1}/{len(gen_text_batches)} sent ({len(audio_int16)} samples)")

        except Exception as e:
            print(f"Streaming error: {e}")
            import traceback
            traceback.print_exc()
            return

    return StreamingResponse(
        audio_generator(), 
        media_type="audio/pcm",
        headers={"X-Sample-Rate": "24000"} 
    )

@app.post("/tts_wav")
async def tts_wav_endpoint(
    text: str = Form(...),
    speed: float = Form(1.0),
    ref_audio_path: Optional[str] = Form(None),
    ref_text: Optional[str] = Form(None),
    ref_audio_file: Optional[UploadFile] = File(None)
):
    """
    Non-streaming endpoint.
    Returns a WAV file directly (audio/wav).
    Accepts reference audio via file upload or path.
    """
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    temp_ref_file = None
    try:
        # 1. Determine Reference Audio Source
        current_ref_audio = REF_AUDIO_PATH # Default
        
        if ref_audio_file:
            # Save uploaded file to temp
            temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            temp_ref_file = temp.name
            temp.close()
            
            with open(temp_ref_file, "wb") as buffer:
                shutil.copyfileobj(ref_audio_file.file, buffer)
            
            current_ref_audio = temp_ref_file
            print(f"[TTS WAV] Using uploaded reference audio: {current_ref_audio}")
            
        elif ref_audio_path:
            current_ref_audio = ref_audio_path
            print(f"[TTS WAV] Using reference audio path: {current_ref_audio}")

        # 2. Determine Reference Text
        current_ref_text = ref_text if ref_text else REF_TEXT
        
        ref_audio_fname, ref_text_final = preprocess_ref_audio_text(
            current_ref_audio, current_ref_text, show_info=lambda x: None
        )
        
        # Load Audio
        audio_np, sr = sf.read(ref_audio_fname)
        if len(audio_np.shape) == 1:
            audio = torch.from_numpy(audio_np).float().unsqueeze(0)
        else:
            audio = torch.from_numpy(audio_np).float().t()
        
        # Parameters
        target_sample_rate = 24000
        hop_length = 256
        target_rms = 0.1
        # Use speed from Form input
        # speed = req.speed if hasattr(req, 'speed') else model.config.speed 
        # Since we are using Form, speed is directly available variable
        
        nfe_step = 8 
        
        # Max chars calculation
        max_chars = int(len(ref_text_final.encode("utf-8")) / (audio.shape[-1] / sr) * (25 - audio.shape[-1] / sr))
        
        # 2. Chunk Input
        gen_text_batches = chunk_text(text, max_chars=max_chars)
        print(f"[TTS WAV] Generating {len(gen_text_batches)} chunks...")
        
        # 3. Process
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        rms = torch.sqrt(torch.mean(torch.square(audio)))
        if rms < target_rms:
            audio = audio * target_rms / rms
        if sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
            audio = resampler(audio)
        
        audio = audio.to(device)
        
        all_chunks = []
        
        for i, gen_text in enumerate(gen_text_batches):
            # Prepare text
            text_list = [ref_text_final + gen_text]
            final_text_list = convert_char_to_pinyin(text_list)
            
            # Duration
            ref_audio_len = audio.shape[-1] // hop_length
            ref_text_len = len(ref_text_final.encode("utf-8"))
            gen_text_len = len(gen_text.encode("utf-8"))
            duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / speed)
            
            # Inference
            with torch.inference_mode():
                generated, _ = model.ema_model.sample(
                    cond=audio,
                    text=final_text_list,
                    duration=duration,
                    steps=nfe_step,
                    cfg_strength=2.0,
                    sway_sampling_coef=-1.0,
                )
                
                generated = generated.to(torch.float32)
                generated = generated[:, ref_audio_len:, :]
                generated_mel_spec = generated.permute(0, 2, 1)
                generated_wave = model.vocoder.decode(generated_mel_spec)
                
                if rms < target_rms:
                    generated_wave = generated_wave * rms / target_rms
                
                generated_wave = generated_wave.squeeze().cpu().numpy()
                audio_int16 = (generated_wave * 32767).astype(np.int16)
                all_chunks.append(audio_int16)

        if not all_chunks:
            raise HTTPException(status_code=400, detail="No audio generated")

        # Combine and Write to WAV
        full_audio = np.concatenate(all_chunks)
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, full_audio, target_sample_rate, format='WAV')
        wav_buffer.seek(0)
        
        print(f"[TTS WAV] Sending {len(full_audio)} samples as WAV")
        return Response(content=wav_buffer.read(), media_type="audio/wav")

    except Exception as e:
        print(f"Error serving WAV: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup temp file if it was created
        if temp_ref_file and os.path.exists(temp_ref_file):
            try:
                os.remove(temp_ref_file)
                print(f"[TTS WAV] Cleaned up temp file: {temp_ref_file}")
            except Exception as cleanup_err:
                print(f"[TTS WAV] Failed to cleanup temp file: {cleanup_err}")

# -----------------------------------------------------------------------------
# 4. FRONTEND
# -----------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>IndicF5 TTS Stream</title>
        <style>
            body { font-family: sans-serif; padding: 20px; max-width: 800px; margin: 0 auto; }
            textarea { width: 100%; padding: 10px; border-radius: 5px; border: 1px solid #ccc; }
            button { padding: 10px 20px; background: #007bff; color: white; border: none; cursor: pointer; border-radius: 5px; margin-top: 10px;}
            button:disabled { background: #ccc; }
            #status { margin-top: 10px; color: #666; }
        </style>
    </head>
    <body>
        <h2>IndicF5 TTS Streaming</h2>
        <p>Enter text (Hindi/Punjabi/English/etc):</p>
        <textarea id="text" rows="5">नमस्ते! आज का दिन बहुत शुभ है। हम नई तकनीकों के बारे में बात कर रहे हैं।</textarea>
        <br>
        <button id="btn" onclick="playAudio()">Generate & Stream</button>
        <div id="status">Ready</div>

        <script>
        async function playAudio() {
            const text = document.getElementById('text').value;
            const btn = document.getElementById('btn');
            const status = document.getElementById('status');
            
            if(!text) return;

            btn.disabled = true;
            status.innerText = "Initializing stream...";

            // 1. Setup Audio Context (24kHz is standard for F5/Vocos)
            const audioCtx = new (window.AudioContext || window.webkitAudioContext)({sampleRate: 24000});
            let nextStartTime = audioCtx.currentTime;
            
            try {
                const response = await fetch('/tts', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({text: text})
                });

                const reader = response.body.getReader();
                let hasStarted = false;

                while (true) {
                    const {done, value} = await reader.read();
                    if (done) break;

                    if(!hasStarted) {
                        status.innerText = "Streaming audio...";
                        hasStarted = true;
                    }

                    // Value is Uint8Array (PCM Bytes)
                    // Convert to Int16
                    const int16Data = new Int16Array(value.buffer, value.byteOffset, value.byteLength / 2);
                    
                    // Convert to Float32 for Web Audio API
                    const float32Data = new Float32Array(int16Data.length);
                    for (let i = 0; i < int16Data.length; i++) {
                        float32Data[i] = int16Data[i] / 32768.0;
                    }

                    // Create Buffer
                    const audioBuffer = audioCtx.createBuffer(1, float32Data.length, 24000);
                    audioBuffer.copyToChannel(float32Data, 0);

                    // Schedule Playback
                    const source = audioCtx.createBufferSource();
                    source.buffer = audioBuffer;
                    source.connect(audioCtx.destination);
                    
                    // Ensure smooth playback without gaps
                    const playTime = Math.max(audioCtx.currentTime, nextStartTime);
                    source.start(playTime);
                    nextStartTime = playTime + audioBuffer.duration;
                }
                
                status.innerText = "Done.";
            } catch (e) {
                console.error(e);
                status.innerText = "Error: " + e.message;
            } finally {
                btn.disabled = false;
            }
        }
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)