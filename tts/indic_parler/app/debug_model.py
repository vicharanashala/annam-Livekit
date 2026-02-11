from parler_tts import ParlerTTSForConditionalGeneration
import torch

try:
    print("Loading model...")
    model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts")
    print("Model loaded.")
    
    print("Audio Encoder Config:", model.audio_encoder.config)
    
    if hasattr(model.audio_encoder.config, "frame_rate"):
        print(f"frame_rate found: {model.audio_encoder.config.frame_rate}")
    else:
        print("frame_rate NOT found in model.audio_encoder.config")
        
    if hasattr(model.audio_encoder.config, "sampling_rate"):
        print(f"sampling_rate found: {model.audio_encoder.config.sampling_rate}")
        
except Exception as e:
    print(f"Error: {e}")
