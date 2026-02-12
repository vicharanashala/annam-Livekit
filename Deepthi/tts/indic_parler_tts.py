import torch
import soundfile as sf
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

# ------------------------------------------------
# 1. Check device (GPU / CPU)
# ------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ------------------------------------------------
# 2. Load model
# ------------------------------------------------
model_name = "ai4bharat/indic-parler-tts"

model = ParlerTTSForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

# ------------------------------------------------
# 3. Load tokenizers
# ------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Description tokenizer (VERY IMPORTANT)
description_tokenizer = AutoTokenizer.from_pretrained(
    model.config.text_encoder._name_or_path
)

# ------------------------------------------------
# 4. Input text (change language freely)
## prompt = "अरे, तुम आज कैसे हो?"   # Hindi
# prompt = (
    "नमस्ते सभी को, आशा है कि आप सभी आज अच्छा महसूस कर रहे होंगे। "
    "यह वॉइस असिस्टेंट स्पष्ट और स्वाभाविक रूप से बोलने के लिए बनाया गया है। "
    "इसे सीखने, कहानियाँ सुनाने और दैनिक बातचीत के लिए उपयोग किया जा सकता है। "
    "सुनने के लिए धन्यवाद, आपका दिन शुभ हो।"
# )

## prompt = "Hello, how are you today?"  # English
# prompt = (
#     "Hello everyone, I hope you are having a wonderful day today. "
#     "This voice assistant is designed to speak clearly and naturally for easy understanding. "
#     "It can be used for learning, storytelling, and daily conversations. "
#     "Thank you for listening, and have a pleasant experience."
# )

# prompt = ( 
#     "ഇത്തരം ടെക്സ്റ്റ്ടു സ്പീച്ച് സിസ്റ്റങ്ങൾ വിദ്യാഭ്യാസം, ആരോഗ്യ മേഖല, "
#     "കസ്റ്റമർ സേവനം, വാർത്താ വായന പോലുള്ള നിരവധി മേഖലകളിൽ "
#     "വളരെ പ്രയോജനകരമാണ്." 
# ) # Malayalam

# prompt = (
#     "ഇത്തരം ടെക്സ്റ്റ് ടു സ്പീച്ച് സിസ്റ്റങ്ങൾ വിദ്യാഭ്യാസ മേഖലയിൽ "
#     "വിദ്യാർത്ഥികൾക്ക് പഠനം കൂടുതൽ എളുപ്പമാക്കാൻ സഹായിക്കുന്നു. "
#     "online classes, digital learning platform, "
#     "മറ്റു എഡ്യൂക്കേഷൻ ആപ്ലിക്കേഷൻസ് എന്നിവയിൽ ഇത് ഉപയോഗിക്കുന്നു."
# ) # tested with different malayalam speakers  output3.wav to output6.wav

prompt = (
    "ഇത്തരം ടെക്സ്റ്റ്-ടു-സ്പീച്ച് സിസ്റ്റങ്ങൾ വിദ്യാഭ്യാസ മേഖലയിലെ "
    "വിദ്യാർത്ഥികൾക്ക് പഠനം കൂടുതൽ എളുപ്പമാക്കാൻ സഹായിക്കുന്നു. "
    "ഓൺലൈൻ-ക്ലാസുകൾ, ഡിജിറ്റൽ-ലേണിംഗ്-പ്ലാറ്റ്-ഫോമുകൾ, "
    "മറ്റ് എഡ്യൂക്കേഷൻ-ആപ്ലിക്കേഷൻുകൾ എന്നിവയിൽ ഇത് ഉപയോഗിക്കുന്നു."
)

prompt = (
    "ഇത്തരം ടെക്സ്റ്റ് ടു സ്പീച്ച് സിസ്റ്റങ്ങൾ വിദ്യാഭ്യാസ മേഖലയിലെ "
    "വിദ്യാർത്ഥികൾക്ക് പഠനം കൂടുതൽ എളുപ്പമാക്കാൻ സഹായിക്കുന്നു. "
    "ഓൺലൈൻ ക്ലാസുകൾ, ഡിജിറ്റൽ ലേണിംഗ് പ്ലാറ്റ്‌ഫോമുകൾ, "
    "മറ്റ് എഡ്യൂക്കേഷൻ ആപ്ലിക്കേഷൻുകൾ എന്നിവയിൽ ഇത് ഉപയോഗിക്കുന്നു."
) #3.1

# prompt = (     
#     "ഓൺലൈൻ ക്ലാസുകൾ, ഡിജിറ്റൽ ലേണിംഗ് പ്ലാറ്റ്‌ഫോമുകൾ, "
#     "മറ്റ് എഡ്യൂക്കേഷൻ ആപ്ലിക്കേഷൻുകൾ എന്നിവയിൽ ഇത് ഉപയോഗിക്കുന്നു."
# ) #output3.2 better


# prompt = (
#     "ഇന്ന് നമ്മൾ ഉപയോഗിക്കുന്ന മൊബൈൽ ആപ്പുകൾയും ഓൺലൈൻ സർവീസുകളും "
#     "ദിവസേനയുള്ള ജീവിതം കൂടുതൽ ലളിതമാക്കുന്നു. "
#     "ഇത്തരം ഡിജിറ്റൽ പ്ലാറ്റ്‌ഫോമുകൾ ആശയവിനിമയം, വിവരശേഖരണം, "
#     "ദൈനംദിന പ്രവർത്തനങ്ങൾ എന്നിവയ്ക്കായി സഹായിക്കുന്നു."
# ) #3.3


# prompt = (
#     "എല്ലാവർക്കും നമസ്കാരം. ഇന്ന് നിങ്ങൾ എല്ലാവരും സന്തോഷത്തോടെയാണെന്ന് ഞാൻ പ്രതീക്ഷിക്കുന്നു. "
#     "ഈ വോയ്സ് അസിസ്റ്റന്റ് വ്യക്തമായും, സ്വാഭാവികമായും സംസാരിക്കാൻ രൂപകൽപ്പന ചെയ്തതാണ്. "   
# ) #3.4

# prompt = (
#     "ഇത്തരം, ടെക്സ്റ്റ് ടു, സ്പീച്ച്, സിസ്റ്റങ്ങൾ വിദ്യാഭ്യാസ മേഖലയിൽ "
#     "വിദ്യാർത്ഥികൾക്ക് പഠനം കൂടുതൽ എളുപ്പമാക്കാൻ സഹായിക്കുന്നു. "     
# ) # 3.5

# prompt = (
#     "ഇത്തരം ടെക്സ്റ്റ് ടു, സ്പീച്ച്, സിസ്റ്റങ്ങൾ വിദ്യാഭ്യാസ മേഖലയിൽ "
#     "വിദ്യാർത്ഥികൾക്ക് പഠനം കൂടുതൽ എളുപ്പമാക്കാൻ സഹായിക്കുന്നു. "     
# ) # 3.6 good

# ------------------------------------------------
# 5. Voice / Style description
# ------------------------------------------------
# description = (
#     "Rohit's voice is clear and very natural, "
#     "slightly expressive with moderate speed and pitch. "
#     "The recording is very clear audio with no background noise."
# )  # Hindi

# description = (
#     "Mary's voice is clear and very natural, "
#     "slightly expressive with moderate speed and pitch. "
#     "The recording is very clear audio with no background noise."
# )  # English


# description = (
#     "Anjali is a native Malayalam speaker speaking in standard Malayalam pronunciation. "
#     "She follows Malayalam speech rhythm strictly, without English-style stress or dramatic intonation. "
#     "English words are pronounced plainly and softly, blended naturally into Malayalam without emphasis. "
#     "The speech is slow, flat, and precise, with clear endings for all words and no syllable merging. "
#     "The recording is extremely close to the microphone with very clear studio-quality audio and zero background noise."
# ) # seems good 

# description = (
#     "Anjali is a native Malayalam speaker using standard Malayalam pronunciation. "
#     "She follows natural Malayalam speech rhythm, without English-style stress or dramatic intonation. "
#     "English words are spoken gently and blend naturally into Malayalam without emphasis. "
#     "The speech is calm and steady, with smooth and complete word endings. "
#     "The recording is clean, clear, and free from background noise."
# ) # seems better
# ===================================================================================================================
# prompt = "ഈ സംവിധാനം പഠനം കൂടുതൽ എളുപ്പം ആക്കാൻ സഹായിക്കുന്നു."
# prompt = "ഈ ടെക്സ്റ്റ് ടു സ്പീച്ച് സംവിധാനം ഓൺലൈൻ പഠനം കൂടുതൽ എളുപ്പമാക്കാൻ സഹായിക്കുന്നു."
# prompt = "ഈ സിസ്റ്റം ഓൺലൈൻ പഠനത്തിന് വളരെ സഹായകരമാണ്."
# prompt = "ആരോഗ്യ മേഖലയിൽ രോഗികൾക്ക് വിവരങ്ങൾ വ്യക്തമായി നൽകാൻ ഇത് സഹായിക്കുന്നു."
# prompt = "ഹെൽത്ത്‌കെയർ ആപ്ലിക്കേഷനുകളിൽ ഈ ടെക്നോളജി ഉപയോഗിക്കുന്നു."
# prompt = "വാർത്തകൾ വ്യക്തമായ ശബ്ദത്തിൽ വായിക്കാൻ ഈ സംവിധാനം സഹായകരമാണ്."
# prompt = "ന്യൂസ് ആപ്ലിക്കേഷനുകളിൽ ഓഡിയോ അപ്‌ഡേറ്റുകൾ നൽകാൻ ഇത് ഉപയോഗിക്കുന്നു."
# prompt = "കൃഷി മേഖലയിൽ കർഷകർക്ക് വിവരങ്ങൾ നൽകാൻ ഈ സംവിധാനം സഹായിക്കുന്നു."
# prompt = "കാഴ്ചവൈകല്യമുള്ള ആളുകൾക്ക് വിവരങ്ങൾ ലഭിക്കാൻ ഇത് സഹായകരമാണ്."
# prompt = "ആക്‌സസിബിലിറ്റി മെച്ചപ്പെടുത്താൻ ഈ ടെക്നോളജി ഉപയോഗിക്കുന്നു."
# output 7.1 to 7.10

description = (
    "Anjali is a native Malayalam speaker using standard Malayalam pronunciation. "
    "She follows natural Malayalam speech rhythm, without English-style stress or dramatic intonation. "
    "English words are spoken gently and blend naturally into Malayalam without emphasis. "
    "The speech is calm and steady, with smooth and complete word endings. "
    "The recording is clean, clear, and free from background noise."
) # seems better
# ===================================================================================================================

# # ------------------------------------------------
# 6. Tokenize inputs
# ------------------------------------------------
description_inputs = description_tokenizer(
    description, return_tensors="pt"
).to(device)

prompt_inputs = tokenizer(
    prompt, return_tensors="pt"
).to(device)

# ------------------------------------------------
# 7. Generate speech
# ------------------------------------------------
with torch.no_grad():
    audio = model.generate(
        input_ids=description_inputs.input_ids,
        attention_mask=description_inputs.attention_mask,
        prompt_input_ids=prompt_inputs.input_ids,
        prompt_attention_mask=prompt_inputs.attention_mask
    )

# ------------------------------------------------
# 8. Save output audio
# ------------------------------------------------
audio = audio.cpu().float().numpy().squeeze()
sf.write("output.wav", audio, model.config.sampling_rate)

print("Speech generated: output.wav")
