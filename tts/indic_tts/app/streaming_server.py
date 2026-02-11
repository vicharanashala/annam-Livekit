"""
Streaming FastAPI Server for Indic TTS
Simulated streaming: generates full audio then streams in chunks
"""

import sys
import time
import threading
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel

# Add parent directory to path to import IndicTTS
sys.path.insert(0, str(Path(__file__).parent.parent))
from inference import IndicTTS, SUPPORTED_LANGUAGES, SPEAKERS

# ------------------------
# Configuration
# ------------------------
CHUNK_SIZE = 4096  # bytes per chunk
SAMPLE_RATE = 22050  # Indic TTS output sample rate

# ------------------------
# Model setup
# ------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model_cache = {}
generation_lock = threading.Lock()

def get_model(lang: str) -> IndicTTS:
    """Get or load model for the specified language."""
    if lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Language '{lang}' not supported. Supported: {list(SUPPORTED_LANGUAGES.keys())}"
        )
        
    if lang not in model_cache:
        print(f"[MODEL] Loading model for language: {lang}")
        model_cache[lang] = IndicTTS(
            model_dir=str((Path(__file__).parent.parent / "models").resolve()),
            lang=lang,
            speaker="female",
            use_cuda=(device == "cuda")
        )
            
    return model_cache[lang]

# ------------------------
# FastAPI app
# ------------------------
app = FastAPI(title="Indic TTS Streaming")

class TTSRequest(BaseModel):
    text: str
    lang: str = "hi"
    speaker: str = "female"

# ------------------------
# Frontend (HTML embedded)
# ------------------------
@app.get("/", response_class=HTMLResponse)
async def frontend():
    return """
<!DOCTYPE html>
<html>
<head>
  <title>Indic TTS Streaming</title>
  <style>
    body { font-family: sans-serif; margin: 2rem; background: #1a1a2e; color: #eee; }
    h2 { color: #00d4ff; }
    textarea { width: 100%; max-width: 600px; padding: 0.5rem; font-size: 1rem; background: #16213e; color: #eee; border: 1px solid #0f3460; }
    select, button { padding: 0.5rem 1rem; margin: 0.5rem 0.5rem 0.5rem 0; font-size: 1rem; cursor: pointer; }
    button { background: #00d4ff; border: none; color: #000; font-weight: bold; }
    button:hover { background: #00a8cc; }
    pre { background: #16213e; padding: 1rem; max-height: 300px; overflow-y: auto; font-size: 0.85rem; }
    .controls { margin: 1rem 0; }
  </style>
</head>
<body>
  <h2>Indic TTS – Streaming Test</h2>

  <textarea id="text" rows="4">नमस्ते, यह एक परीक्षण है। यह स्ट्रीमिंग टेक्स्ट टू स्पीच है।</textarea>

  <div class="controls">
    <select id="lang">
      <option value="hi" selected>Hindi (hi)</option>
      <option value="ta">Tamil (ta)</option>
      <option value="te">Telugu (te)</option>
      <option value="bn">Bengali (bn)</option>
      <option value="mr">Marathi (mr)</option>
      <option value="gu">Gujarati (gu)</option>
      <option value="kn">Kannada (kn)</option>
      <option value="ml">Malayalam (ml)</option>
      <option value="or">Odia (or)</option>
      <option value="as">Assamese (as)</option>
    </select>
    <select id="speaker">
      <option value="female" selected>Female</option>
      <option value="male">Male</option>
    </select>
    <button onclick="startTTS()">Start Streaming</button>
  </div>

  <pre id="log"></pre>

<script>
async function startTTS() {
  const log = msg => {
    document.getElementById("log").textContent += msg + "\\n";
    console.log(msg);
  };

  document.getElementById("log").textContent = "";

  const ctx = new AudioContext({ sampleRate: 22050 });
  await ctx.resume();
  
  let started = false;
  let nextPlayTime = ctx.currentTime;
  const t0 = performance.now();

  log("Sending request...");

  try {
    const res = await fetch("/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: document.getElementById("text").value,
        lang: document.getElementById("lang").value,
        speaker: document.getElementById("speaker").value
      })
    });

    if (!res.ok) {
      const errText = await res.text();
      log(`Error: ${res.status} ${res.statusText} - ${errText}`);
      return;
    }

    const reader = res.body.getReader();
    let pending = new Uint8Array(0);

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      if (!started) {
        log(`First audio chunk at ${(performance.now() - t0).toFixed(1)} ms`);
        started = true;
      }

      // Concatenate with pending bytes
      const chunk = new Uint8Array(pending.length + value.length);
      chunk.set(pending);
      chunk.set(value, pending.length);

      // Determine complete samples (2 bytes per sample for Int16)
      const sampleCount = Math.floor(chunk.length / 2);
      const alignedLength = sampleCount * 2;

      // Extract complete PCM data
      const pcmData = chunk.subarray(0, alignedLength);
      
      // Save remainder for next iteration
      pending = chunk.slice(alignedLength);

      if (pcmData.length === 0) continue;

      // Create Int16Array from the aligned data
      const pcm16 = new Int16Array(pcmData.buffer.slice(pcmData.byteOffset, pcmData.byteOffset + pcmData.byteLength));
      
      const audioBuffer = ctx.createBuffer(1, pcm16.length, 22050);
      const channel = audioBuffer.getChannelData(0);

      for (let i = 0; i < pcm16.length; i++) {
        channel[i] = pcm16[i] / 32768;
      }

      const src = ctx.createBufferSource();
      src.buffer = audioBuffer;
      src.connect(ctx.destination);
      
      const playTime = Math.max(ctx.currentTime, nextPlayTime);
      src.start(playTime);
      nextPlayTime = playTime + audioBuffer.duration;
      
      log(`Chunk: ${pcm16.length} samples, duration: ${audioBuffer.duration.toFixed(3)}s`);
    }

    log(`Streaming finished in ${(performance.now() - t0).toFixed(1)} ms`);
  } catch (error) {
    log(`Error: ${error.message}`);
    console.error(error);
  }
}
</script>
</body>
</html>
"""

# ------------------------
# Health check
# ------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "loaded_models": list(model_cache.keys())}

@app.get("/languages")
async def languages():
    return {"languages": SUPPORTED_LANGUAGES, "speakers": SPEAKERS}

# ------------------------
# Streaming TTS endpoint
# ------------------------
@app.post("/tts")
async def tts(req: TTSRequest):
    start = time.time()
    print(f"[REQ] text={req.text[:50]}... lang={req.lang} speaker={req.speaker}")

    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
        
    if req.speaker not in SPEAKERS:
        raise HTTPException(status_code=400, detail=f"Speaker '{req.speaker}' not supported. Supported: {SPEAKERS}")

    # Get model
    tts_model = get_model(req.lang)

    def stream():
        first = True
        try:
            with generation_lock:
                # Generate full audio
                wav, sample_rate = tts_model.synthesize(
                    text=req.text,
                    speaker=req.speaker
                )
            
            gen_time = time.time() - start
            print(f"[TTS] Generated in {gen_time:.3f}s, {len(wav)} samples")

            # Convert to int16 PCM
            if isinstance(wav, np.ndarray):
                audio_int16 = (wav * 32767).astype(np.int16)
            else:
                audio_int16 = (np.array(wav) * 32767).astype(np.int16)
            
            audio_bytes = audio_int16.tobytes()

            # Stream in chunks
            offset = 0
            while offset < len(audio_bytes):
                chunk = audio_bytes[offset:offset + CHUNK_SIZE]
                if first:
                    print(f"[TTS] First chunk at {time.time() - start:.3f}s")
                    first = False
                yield chunk
                offset += CHUNK_SIZE

            print(f"[DONE] Total {time.time() - start:.3f}s")

        except Exception as e:
            print(f"[ERROR] {e}")
            raise

    return StreamingResponse(stream(), media_type="audio/pcm")

# ------------------------
# Run
# ------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9028)
