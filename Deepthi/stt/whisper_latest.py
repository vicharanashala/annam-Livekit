import whisper

# Load latest high-accuracy model
model = whisper.load_model("large-v3")

# Transcribe audio
result = model.transcribe(
    "/home/aic_u3/aic_u3/Speech_Models/sound1_female.wav",
    language="ml",
    task="transcribe"
)

print("Detected Language:", result["language"])
print("Transcription:\n", result["text"])

for segment in result["segments"]:
    print(f"[{segment['start']:.2f} - {segment['end']:.2f}] {segment['text']}")

# Save clean TXT instead of JSON
with open("whisper_output.txt", "w", encoding="utf-8") as f:
    f.write("Detected Language: " + result["language"] + "\n\n")
    f.write("Transcription:\n")
    f.write(result["text"].strip() + "\n\n")
    
    f.write("Timestamps:\n")
    for segment in result["segments"]:
        start = f"{segment['start']:.2f}"
        end = f"{segment['end']:.2f}"
        text = segment["text"].strip()
        f.write(f"[{start} - {end}] {text}\n")

print("✅ Saved output to whisper_output.txt")
