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
        text="অসমীয়া সাহিত্যত লোককথাৰ পৰা আধুনিক গল্পলৈকে চহকী পৰম্পৰা আছে",
        ref_audio_path="voicewav2.wav",
        ref_text="অসমীয়া সাহিত্যত লোককথাৰ পৰা আধুনিক গল্পলৈকে চহকী পৰম্পৰা আছে"
    )