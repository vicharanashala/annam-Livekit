from transformers import AutoProcessor, AutoModelForCTC
import torchaudio
import torch

model_id = "facebook/mms-1b-all"

print("Loading model...")
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForCTC.from_pretrained(model_id)

# ✅ Set language
processor.tokenizer.set_target_lang("mal")

# ✅ Load Malayalam adapter
model.load_adapter("mal")

audio_path = "/home/aic_u3/aic_u3/Speech_Models/sound9_male.wav"

print("Loading audio...")
wav, sr = torchaudio.load(audio_path)

if wav.shape[0] > 1:
    wav = torch.mean(wav, dim=0)
else:
    wav = wav.squeeze()

if sr != 16000:
    print(f"Resampling from {sr} to 16000")
    resampler = torchaudio.transforms.Resample(sr, 16000)
    wav = resampler(wav)

print("Processing audio...")

inputs = processor(
    wav,
    sampling_rate=16000,
    return_tensors="pt"
)

with torch.no_grad():
    logits = model(**inputs).logits

predicted_ids = torch.argmax(logits, dim=-1)

transcription = processor.batch_decode(predicted_ids)[0]

print("\n--- TRANSCRIPTION ---\n")
print(transcription)

with open("wav2vec_output.txt", "w", encoding="utf-8") as f:
    f.write(transcription)

print("\nSaved output to wav2vec_output.txt")
