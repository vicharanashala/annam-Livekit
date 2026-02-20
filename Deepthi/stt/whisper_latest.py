# Model1-  large-v3
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

# Model2.......................................................
# # model_id = "openai/whisper-large-v3-turbo"

# import torch
# from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# # -------------------------------------------------
# # 1. Model Setup
# # -------------------------------------------------
# model_id = "openai/whisper-large-v3-turbo"

# device = "cuda" if torch.cuda.is_available() else "cpu"
# torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# model = AutoModelForSpeechSeq2Seq.from_pretrained(
#     model_id,
#     torch_dtype=torch_dtype,
#     low_cpu_mem_usage=True,
#     use_safetensors=True
# ).to(device)

# processor = AutoProcessor.from_pretrained(model_id)

# # Force Malayalam correctly
# forced_decoder_ids = processor.get_decoder_prompt_ids(
#     language="ml",
#     task="transcribe"
# )

# pipe = pipeline(
#     "automatic-speech-recognition",
#     model=model,
#     tokenizer=processor.tokenizer,
#     feature_extractor=processor.feature_extractor,
#     torch_dtype=torch_dtype,
#     device=device,
# )

# # -------------------------------------------------
# # 2. Transcription (Stable decoding settings)
# # -------------------------------------------------
# result = pipe(
#     "/home/aic_u3/aic_u3/Speech_Models/sound5_male.wav",
#     return_timestamps=True,
#     chunk_length_s=20,   # break long audio (VERY IMPORTANT)
#     generate_kwargs={
#         "forced_decoder_ids": forced_decoder_ids,
#         "temperature": 0.0,
#         "do_sample": False,
#         "repetition_penalty": 1.2,   # prevents ററററ loops
#         "no_repeat_ngram_size": 3,
#         "num_beams": 1,
#     },
# )

# print("\nTranscription:\n", result["text"])

# # -------------------------------------------------
# # 3. Safe Timestamp Writing
# # -------------------------------------------------
# with open("whisper_turbo.txt", "w", encoding="utf-8") as f:
#     f.write("Model: openai/whisper-large-v3-turbo\n\n")
#     f.write("Transcription:\n")
#     f.write(result["text"].strip() + "\n\n")
#     f.write("Timestamps:\n")

#     for chunk in result["chunks"]:
#         start = chunk["timestamp"][0]
#         end = chunk["timestamp"][1]

#         # Handle None safely
#         if start is None:
#             start = 0.0
#         if end is None:
#             end = start

#         text = chunk["text"].strip()

#         f.write(f"[{start:.2f} - {end:.2f}] {text}\n")

# print("✅ Saved output to whisper_turbo.txt")
