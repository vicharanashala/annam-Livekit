from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
from parler_tts import ParlerTTSConfig
import soundfile as sf
import torch
import numpy as np
import os

device = "cuda:0"

checkpoints = [
    "checkpoint-500-epoch-3",
    "checkpoint-1000-epoch-6",
    "checkpoint-1500-epoch-10",
    "checkpoint-2000-epoch-13",
    "checkpoint-2500-epoch-17",
    "checkpoint-3000-epoch-20",
    "checkpoint-3500-epoch-24",
    "checkpoint-4000-epoch-27",
    "checkpoint-4290-epoch-29"
]

base_path = "/home/jupyter-koustav/fine_tuning/punjabi_parler_output"
description = "A female speaker speaks clearly in Punjabi with a natural tone and moderate pace in a quiet environment."
prompt = "ripa karaṃṭa samuṃdarī taṭṭa nūṃ tor̤adīāṃ lahirāṃ toṃ vāpasa āuṇa vālā vahāa huṃdā hai"

tokenizer = AutoTokenizer.from_pretrained(base_path)

# load config once from base path
config = ParlerTTSConfig.from_pretrained(base_path)

for checkpoint in checkpoints:
    print(f"\nRunning inference on {checkpoint}...")
    
    checkpoint_path = f"{base_path}/{checkpoint}"
    print(f"Files in checkpoint: {os.listdir(checkpoint_path)}")
    
    # load model with base config but checkpoint weights
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        checkpoint_path,
        config=config
    ).to(device)
    
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
            prompt_attention_mask=prompt_attention_mask,
            do_sample=True,
            temperature=0.7,
            min_new_tokens=10,
            max_length=2580
        )
    
    audio = generation.cpu().numpy().squeeze()
    audio = audio / np.max(np.abs(audio))
    audio = (audio * 32767).astype(np.int16)
    
    output_path = f"/home/jupyter-koustav/fine_tuning/{checkpoint}_output.wav"
    sf.write(output_path, audio, model.config.sampling_rate)
    print(f"Saved: {output_path}")
    
    del model
    torch.cuda.empty_cache()

print("\nAll checkpoints done!")