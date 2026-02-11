import time
import torch
import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse, Response
from pydantic import BaseModel
from transformers import AutoModel
import os
import sys

# Ensure we can import f5_tts from local repo if needed
# Assuming structure: .../indicf5/app/main.py -> .../indicf5/IndicF5
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDIC_F5_PATH = os.path.join(BASE_DIR, "IndicF5")
if INDIC_F5_PATH not in sys.path:
    sys.path.append(INDIC_F5_PATH)

try:
    from f5_tts.infer.utils_infer import infer_process, chunk_text, preprocess_ref_audio_text
    from f5_tts.model.utils import convert_char_to_pinyin
except ImportError:
    print("WARNING: Could not import infer_process directly. Make sure f5_tts is in python path.")

import torchaudio
import soundfile as sf




# ------------------------
# Model setup
# ------------------------
repo_id = "ai4bharat/IndicF5"
print(f"Loading model from {repo_id}...")

# Load model with trust_remote_code=True as requested
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
model = model.to(device)

# Reference audio setup
# Calculating absolute path to the prompt file
# Structure:
# .../indicf5/app/main.py
# .../indicf5/IndicF5/prompts/PAN_F_HAPPY_00001.wav
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_AUDIO_PATH = os.path.join(BASE_DIR, "IndicF5/prompts/PAN_F_HAPPY_00001.wav")
REF_TEXT = "ਭਹੰਪੀ ਵਿੱਚ ਸਮਾਰਕਾਂ ਦੇ ਭਵਨ ਨਿਰਮਾਣ ਕਲਾ ਦੇ ਵੇਰਵੇ ਗੁੰਝਲਦਾਰ ਅਤੇ ਹੈਰਾਨ ਕਰਨ ਵਾਲੇ ਹਨ, ਜੋ ਮੈਨੂੰ ਖੁਸ਼ ਕਰਦੇ  ਹਨ।"

if not os.path.exists(REF_AUDIO_PATH):
    print(f"WARNING: Reference audio file not found at {REF_AUDIO_PATH}")
else:
    print(f"Reference audio found: {REF_AUDIO_PATH}")

# ------------------------
# FastAPI app
# ------------------------
app = FastAPI(title="IndicF5 TTS Streaming")

class TTSRequest(BaseModel):
    text: str

# ------------------------
# Favicon (silence 404s)
# ------------------------
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

# ------------------------
# Frontend (HTML embedded)
# ------------------------
@app.get("/", response_class=HTMLResponse)
async def frontend():
    return """
<!DOCTYPE html>
<html>
<head>
  <title>IndicF5 TTS Streaming</title>
</head>
<body>
  <h2>IndicF5 TTS – Streaming Test</h2>

  <textarea id="text" rows="4" cols="70">
नमस्ते! संगीत की तरह जीवन भी खूबसूरत होता है, बस इसे सही ताल में जीना आना चाहिए.
  </textarea><br><br>

  <button onclick="startTTS()">Start Streaming</button>

  <pre id="log"></pre>

<script>
async function startTTS() {
  const log = msg => {
    document.getElementById("log").textContent += msg + "\\n";
    console.log(msg);
  };

  // Sample rate for IndicF5 is 24000
  const ctx = new AudioContext({ sampleRate: 24000 });
  await ctx.resume(); 
  
  let started = false;
  let nextPlayTime = ctx.currentTime;
  const t0 = performance.now();

  log("Sending request...");

  try {
    const res = await fetch("/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: document.getElementById("text").value })
    });

    if (!res.ok) {
      log(`Error: ${res.status} ${res.statusText}`);
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

      const chunk = new Uint8Array(pending.length + value.length);
      chunk.set(pending);
      chunk.set(value, pending.length);

      // Assuming Int16 (2 bytes per sample)
      const sampleCount = Math.floor(chunk.length / 2);
      const alignedLength = sampleCount * 2;
      const remainder = chunk.length % 2;

      const pcmData = chunk.subarray(0, alignedLength);
      pending = chunk.slice(alignedLength);

      if (pcmData.length === 0) continue;

      const pcm16 = new Int16Array(pcmData.buffer.slice(pcmData.byteOffset, pcmData.byteOffset + pcmData.byteLength));
      
      const audioBuffer = ctx.createBuffer(1, pcm16.length, 24000);
      const channel = audioBuffer.getChannelData(0);

      // Convert Int16 to Float32 [-1.0, 1.0]
      for (let i = 0; i < pcm16.length; i++) {
        channel[i] = pcm16[i] / 32768;
      }

      const src = ctx.createBufferSource();
      src.buffer = audioBuffer;
      src.connect(ctx.destination);
      
      const playTime = Math.max(ctx.currentTime, nextPlayTime);
      src.start(playTime);
      nextPlayTime = playTime + audioBuffer.duration;
      
      log(`Chunk queued: ${pcm16.length} samples, duration: ${audioBuffer.duration.toFixed(3)}s`);
    }

    log("Streaming finished");
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
# Streaming TTS endpoint
# ------------------------
@app.post("/tts")
async def tts(req: TTSRequest):
    start = time.time()
    print(f"[REQ] received: {req.text[:50]}...")

    # Define streaming generator
    def audio_stream():
        # 1. Prepare Reference Audio
        # We use strict cache/optimize later, for now follow standard flow
        try:
             # Preprocess ref audio (trim silence etc)
             # Note: This writes to temp file, might be slow if repeated. 
             # Ideally we cache "ref_audio_tensor".
             ref_audio_fname, ref_text_final = preprocess_ref_audio_text(REF_AUDIO_PATH, REF_TEXT, show_info=lambda x: None)
             
             # Load Audio Tensor
             # audio, sr = torchaudio.load(ref_audio_fname)
             # FIX: Use soundfile to avoid TorchCodec errors
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
             cross_fade_duration = 0.15 
             speed = model.config.speed
             nfe_step = 8 # User requested optimization

             # Max chars calculation (same as Infer Process)
             max_chars = int(len(ref_text_final.encode("utf-8")) / (audio.shape[-1] / sr) * (25 - audio.shape[-1] / sr))
             
             # 2. Chunk the Input Text
             gen_text_batches = chunk_text(req.text, max_chars=max_chars)
             print(f"[TTS] Chunked into {len(gen_text_batches)} parts.")

             # 3. Process Logic
             if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
             
             rms = torch.sqrt(torch.mean(torch.square(audio)))
             if rms < target_rms:
                audio = audio * target_rms / rms
             if sr != target_sample_rate:
                resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
                audio = resampler(audio)
             
             audio = audio.to(device) # Cond Audio
             
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
                     
                     # Simple yield (No cross-fade for streaming simplicity/speed)
                     # Convert to Int16 PCM
                     audio_int16 = (generated_wave * 32767).astype(np.int16)
                     yield audio_int16.tobytes()
                     
                     if first_chunk:
                         print(f"[TTS] First chunk ready in {time.time() - start:.3f}s")
                         first_chunk = False
                     print(f"[TTS] Chunk {i+1}/{len(gen_text_batches)} sent ({len(audio_int16)} samples)")

        except Exception as e:
            print(f"Error in stream: {e}")
            import traceback
            traceback.print_exc()

    return StreamingResponse(audio_stream(), media_type="audio/pcm")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9026)
