from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import torch
import numpy as np

device = "cpu"
model = ParlerTTSForConditionalGeneration.from_pretrained(
    "/home/jupyter-koustav/fine_tuning/parler_output"
).to(device)
tokenizer = AutoTokenizer.from_pretrained(
    "/home/jupyter-koustav/fine_tuning/parler_output"
)

description = "A male speaker speaks clearly in English with a natural tone and moderate pace in a quiet environment."
prompt = "Hello, this is a test of the fine-tuned model."

desc = tokenizer(description, return_tensors="pt")
prompt_tokens = tokenizer(prompt, return_tensors="pt")

input_ids = desc.input_ids.to(device)
attention_mask = desc.attention_mask.to(device)
prompt_input_ids = prompt_tokens.input_ids.to(device)
prompt_attention_mask = prompt_tokens.attention_mask.to(device)

with torch.no_grad():
    generation = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_input_ids=prompt_input_ids,
        prompt_attention_mask=prompt_attention_mask
    )

audio = generation.cpu().numpy().squeeze()
audio = audio / np.max(np.abs(audio))
sf.write(
    "/home/jupyter-koustav/fine_tuning/test_output_fixed.wav",
    audio,
    model.config.sampling_rate
)
print("Audio saved")