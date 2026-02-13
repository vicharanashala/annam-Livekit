# ---------------------------------------------------------
# Modified scripts 

import requests

def generate_tts_wav(text: str, ref_audio_path: str, ref_text: str, output_file: str = "upload_test.wav"):
    url = "https://mesne-unlicentiously-allie.ngrok-free.dev/tts_wav"

    # Form fields (same as -F in curl)
    data = {
        "text": text,
        "ref_text": ref_text
    }

    # File field
    files = {
        "ref_audio_file": open(ref_audio_path, "rb")
    }

    try:
        response = requests.post(url, data=data, files=files)

        if response.status_code == 200:
            with open(output_file, "wb") as f:
                f.write(response.content)
            print(f"Audio saved as {output_file}")
        else:
            print("Request failed:", response.status_code)
            print(response.text)

    finally:
        files["ref_audio_file"].close()


if __name__ == "__main__":
    generate_tts_wav(
        text="""
        ഇത്തരം ടെക്സ്റ്റ്-ടു-സ്പീച്ച് സിസ്റ്റങ്ങൾ വിദ്യാഭ്യാസ മേഖലയിലെ 
        വിദ്യാർത്ഥികൾക്ക് പഠനം കൂടുതൽ എളുപ്പമാക്കാൻ സഹായിക്കുന്നു. 
        ഓൺലൈൻ-ക്ലാസുകൾ, ഡിജിറ്റൽ-ലേണിംഗ്-പ്ലാറ്റ്-ഫോമുകൾ, 
        മറ്റ് എഡ്യൂക്കേഷൻ-ആപ്ലിക്കേഷൻുകൾ എന്നിവയിൽ ഇത് ഉപയോഗിക്കുന്നു.
        """,
        ref_audio_path="/home/aic_u3/aic_u3/Speech_Models/sound4_male.wav",
        # ref_text="ഇന്ന് കാലാവസ്ഥ വളരെ നല്ലതാണ്.",
        # ref_text="ഇന്ന് കാലാവസ്ഥ വളരെ നല്ലതാണ്. അതിനാൽ എനിക്ക് ഇന്ന് വളരെ സന്തോഷം ആണ്",
        ref_text="എനിക്ക് ഇന്ന് വളരെ സന്തോഷം ആണ്. ഇന്ന് കാലാവസ്ഥ വളരെ നല്ലതാണ്. ഞാൻ ശാന്തമായി സംസാരിക്കുന്നു.",
        output_file="/home/aic_u3/aic_u3/Speech_Models/generated_audio/upload_test8.wav"
    )