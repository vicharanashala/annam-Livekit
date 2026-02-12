from transformers import AutoModel
import numpy as np
import soundfile as sf

# Load IndicF5 from Hugging Face
repo_id = "ai4bharat/IndicF5"
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)
model = model.to("cuda")
model.eval()

# TRANSCRIPT OF YOUR RECORDED AUDIO
# reference_text = "എനിക്ക് ഇന്ന് വളരെ സന്തോഷം ആണ്." #sound1.wav
# reference_text = "ഇന്ന് കാലാവസ്ഥ വളരെ നല്ലതാണ്." #sound2.wav
# reference_text = "ഇന്ന് കാലാവസ്ഥ വളരെ നല്ലതാണ്. അതിനാൽ എനിക്ക് ഇന്ന് വളരെ സന്തോഷം ആണ്." #sound3.wav
# reference_text = "എനിക്ക് ഇന്ന് വളരെ സന്തോഷം ആണ്. ഇന്ന് കാലാവസ്ഥ വളരെ നല്ലതാണ്. ഞാൻ ശാന്തമായി സംസാരിക്കുന്നു." #sound4.wav
# Generate speech
# audio = model(
#     # "നമസ്കാരം, ഇന്ന് നിങ്ങൾക്ക് എങ്ങനെയുണ്ട്?",
#     "നമസ്കാരം. ഇന്ന് നിങ്ങൾക്ക് എങ്ങനെയുണ്ട്? ഞാൻ നിങ്ങളോട് സംസാരിക്കുന്നത് വളരെ സന്തോഷത്തോടെ ആണ്.",
#     ref_audio_path="sound4.wav",
#     ref_text="എനിക്ക് ഇന്ന് വളരെ സന്തോഷം ആണ്. ഇന്ന് കാലാവസ്ഥ വളരെ നല്ലതാണ്. ഞാൻ ശാന്തമായി സംസാരിക്കുന്നു."
# )

# audio = model(
#     "आज मैंने एक नई चीज़ सीखी",
#     ref_audio_path="hindi1.wav",
#     ref_text="नमस्ते, मेरा नाम दीप्ति है। आज का दिन बहुत अच्छा है और मैं बहुत खुश महसूस कर रही हूँ।"
# )

# audio = model(
#     "Artificial intelligence is transforming the world in amazing ways.",
#     ref_audio_path="english1.wav",
#     ref_text="Hello, my name is Kajal. Today is a beautiful day and I am feeling very happy. I enjoy learning new technologies and working on interesting projects."
# )
# Used github audio for reference
# audio = model(
#     "आज मैंने एक नई चीज़ सीखी",
#     ref_audio_path="PAN_F_HAPPY_00001.wav",
#     ref_text="ਭਹੰਪੀ ਵਿੱਚ ਸਮਾਰਕਾਂ ਦੇ ਭਵਨ ਨਿਰਮਾਣ ਕਲਾ ਦੇ ਵੇਰਵੇ ਗੁੰਝਲਦਾਰ ਅਤੇ ਹੈਰਾਨ ਕਰਨ ਵਾਲੇ ਹਨ, ਜੋ ਮੈਨੂੰ ਖੁਸ਼ ਕਰਦੇ  ਹਨ।"
# )
# used generate_tts_wav.py to generate audio for corresponding text reference
audio = model(
    "എനിക്ക് ഇന്ന് വളരെ സന്തോഷം ആണ്",
    ref_audio_path="my_audio.wav",
    # ref_text="नमस्ते, मेरा नाम दीप्ति है। आज का दिन बहुत अच्छा है और मैं बहुत खुश महसूस कर रही हूँ।",
    ref_text="എനിക്ക് ഇന്ന് വളരെ സന്തോഷം ആണ്. ഇന്ന് കാലാവസ്ഥ വളരെ നല്ലതാണ്. ഞാൻ ശാന്തമായി സംസാരിക്കുന്നു"
)

# Normalize and save output
if audio.dtype == np.int16:
    audio = audio.astype(np.float32) / 32768.0
sf.write("namaste.wav", np.array(audio, dtype=np.float32), samplerate=24000)
print("Audio saved succesfully.")


import soundfile as sf

data, sr = sf.read("my_audio.wav")
print("Sample rate:", sr)
print("Duration:", len(data)/sr, "seconds")


