from transformers import AutoModel
import torch, torchaudio

# Load the model
model = AutoModel.from_pretrained("ai4bharat/indic-conformer-600m-multilingual", trust_remote_code=True)

# Load an audio file
wav, sr = torchaudio.load("/home/aic_u3/aic_u3/Speech_Models/sound9_female.wav")
wav = torch.mean(wav, dim=0, keepdim=True)

target_sample_rate = 16000  # Expected sample rate
if sr != target_sample_rate:
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sample_rate)
    wav = resampler(wav)

# Perform ASR with CTC decoding
transcription_ctc = model(wav, "ml", "ctc")
print("CTC Transcription:", transcription_ctc)

# Perform ASR with RNNT decoding
transcription_rnnt = model(wav, "ml", "rnnt")
print("RNNT Transcription:", transcription_rnnt)

# ....................
# -------------------------------
# Final Clean Output Section
# -------------------------------

print("\n" + "="*50)
print("FINAL CLEAN OUTPUT")
print("="*50)

# Print using repr (sometimes cleaner in terminals)
print("CTC (repr):", repr(transcription_ctc))
print("RNNT (repr):", repr(transcription_rnnt))

# Save to file with UTF-8 encoding
with open("final_transcription9.txt", "w", encoding="utf-8") as f:
    f.write("CTC Transcription:\n")
    f.write(transcription_ctc + "\n\n")
    f.write("RNNT Transcription:\n")
    f.write(transcription_rnnt + "\n")

print("\nSaved clean output to: final_transcription9.txt")
print("="*50)

# ....................................................
