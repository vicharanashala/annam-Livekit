import time
import threading
import torch
import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
from parler_tts import ParlerTTSForConditionalGeneration, ParlerTTSStreamer
from transformers import AutoTokenizer

# ------------------------
# Model setup
# ------------------------
device = "cuda" 
MODEL_ID = "ai4bharat/indic-parler-tts"
model = ParlerTTSForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="cuda" if device == "cuda" else None
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# Description tokenizer for voice characteristics
description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
generation_lock = threading.Lock()

# Default voice description
DEFAULT_DESCRIPTION = "A clear voice with a moderate pace and pitch. The recording is of high quality with no background noise."

# ------------------------
# FastAPI app
# ------------------------
app = FastAPI(title="Parler TTS Streaming (All-in-One)")

class TTSRequest(BaseModel):
    text: str
    description: str = None  # Optional: custom voice description

def locked_generate(**kwargs):
    with generation_lock:
        model.generate(**kwargs)

# ------------------------
# Frontend (HTML embedded)
# ------------------------
@app.get("/", response_class=HTMLResponse)
async def frontend():
    return """
<!DOCTYPE html>
<html>
<head>
  <title>Parler TTS Streaming</title>
</head>
<body>
  <h2>Parler TTS – Streaming Test</h2>

  <textarea id="text" rows="4" cols="70">
Hello, this is real time text to speech streaming.
  </textarea><br><br>

  <button onclick="startTTS()">Start Streaming</button>

  <pre id="log"></pre>

<script>
async function startTTS() {
  const log = msg => {
    document.getElementById("log").textContent += msg + "\\n";
    console.log(msg);
  };

  const ctx = new AudioContext({ sampleRate: 44100 });
  await ctx.resume(); // Ensure AudioContext is running
  
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

      // Concatenate with pending bytes
      const chunk = new Uint8Array(pending.length + value.length);
      chunk.set(pending);
      chunk.set(value, pending.length);

      // Determine complete samples
      const sampleCount = Math.floor(chunk.length / 2);
      const alignedLength = sampleCount * 2;
      const remainder = chunk.length % 2;

      // Extract complete PCM data
      const pcmData = chunk.subarray(0, alignedLength);
      
      // Save remainder for next iteration
      pending = chunk.slice(alignedLength);

      if (pcmData.length === 0) continue;

      // Create Int16Array from the aligned data
      // We must copy the buffer to ensure byteOffset is 0 and it's safe to use
      const pcm16 = new Int16Array(pcmData.buffer.slice(pcmData.byteOffset, pcmData.byteOffset + pcmData.byteLength));
      
      const audioBuffer = ctx.createBuffer(1, pcm16.length, 44100);
      const channel = audioBuffer.getChannelData(0);

      for (let i = 0; i < pcm16.length; i++) {
        channel[i] = pcm16[i] / 32768;
      }

      const src = ctx.createBufferSource();
      src.buffer = audioBuffer;
      src.connect(ctx.destination);
      
      // Schedule playback at the correct time
      const playTime = Math.max(ctx.currentTime, nextPlayTime);
      src.start(playTime);
      
      // Update next play time
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
    print("[REQ] received")

    # Tokenize description (voice characteristics) with description_tokenizer
    voice_description = req.description if req.description else DEFAULT_DESCRIPTION
    description_inputs = description_tokenizer(
        voice_description, 
        return_tensors="pt"
    ).to(device)
    
    # Tokenize prompt (text to speak) with main tokenizer
    prompt_inputs = tokenizer(
        req.text, 
        return_tensors="pt"
    ).to(device)

    sampling_rate = model.audio_encoder.config.sampling_rate

    # ==========================================
    # OPTION 1: OLD STYLE (Currently Active)
    # ==========================================
    # play_steps = int(sampling_rate * 0.35)  # ~350 ms chunks
    frame_rate = getattr(model.audio_encoder.config, "frame_rate", 86)
    play_steps = int(frame_rate * 3) # 0.5s chunks
    print("play_steps", play_steps)
    print("play _steps_old",int(sampling_rate * 0.35))
    streamer = ParlerTTSStreamer(
        model,
        device=device,
        play_steps=play_steps
    )

    gen_kwargs = dict(
        input_ids=description_inputs.input_ids,
        attention_mask=description_inputs.attention_mask,
        prompt_input_ids=prompt_inputs.input_ids,
        prompt_attention_mask=prompt_inputs.attention_mask,
        streamer=streamer
    )

    # ==========================================
    # OPTION 2: NEW STYLE (Commented Out)
    # ==========================================
    # frame_rate = getattr(model.audio_encoder.config, "frame_rate", 86)
    # play_steps = int(frame_rate * 0.5) # 0.5s chunks
    # 
    # streamer = ParlerTTSStreamer(
    #     model,
    #     device=device,
    #     play_steps=play_steps
    # )
    # 
    # gen_kwargs = dict(
    #     input_ids=description_inputs.input_ids,
    #     attention_mask=description_inputs.attention_mask,
    #     prompt_input_ids=prompt_inputs.input_ids,
    #     prompt_attention_mask=prompt_inputs.attention_mask,
    #     streamer=streamer,
    #     do_sample=True,
    #     temperature=1.0,
    #     min_new_tokens=10,
    # )
    # ==========================================
    
    thread = threading.Thread(target=locked_generate, kwargs=gen_kwargs)
    thread.start()

    def stream():
        first = True
        for chunk in streamer:
            if chunk.shape[0] == 0:
                break
            
            now = time.time()
            if first:
                print(f"[TTS] first chunk in {now - start:.3f}s")
                first = False

            # Convert to numpy if it's a tensor, otherwise use directly
            if isinstance(chunk, torch.Tensor):
                audio = (chunk.cpu().numpy() * 32767).astype(np.int16)
            else:
                audio = (chunk * 32767).astype(np.int16)
            
            # Ensure even number of samples for Int16Array compatibility
            if len(audio) % 2 != 0:
                audio = audio[:-1]
            
            print(f"[TTS] chunk @ {now - start:.3f}s | {len(audio)} samples")
            yield audio.tobytes()

        print(f"[DONE] total {time.time() - start:.3f}s")

    return StreamingResponse(stream(), media_type="audio/pcm")

# ------------------------
# Run
# ------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9026)
