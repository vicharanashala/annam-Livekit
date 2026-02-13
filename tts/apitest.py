import requests
import time
import os
import csv

def generate_tts_wav():

    url = "https://mesne-unlicentiously-allie.ngrok-free.dev/tts_wav"

    output_folder = "tts_outputs"
    os.makedirs(output_folder, exist_ok=True)

    texts = {
        "hindi": "आज मैंने एक नई चीज़ सीखी।",
        "tamil": "இன்று நான் ஒரு புதிய விஷயத்தை கற்றுக்கொண்டேன்.",
        "telugu": "ఈ రోజు నేను ఒక కొత్త విషయం నేర్చుకున్నాను.",
        "bengali": "আজ আমি একটি নতুন জিনিস শিখেছি।",
        "assamese": "আজি মই এটা নতুন বস্তু শিকিলোঁ।",
        "kannada": "ಇಂದು ನಾನು ಒಂದು ಹೊಸ ವಿಷಯವನ್ನು ಕಲಿತೆ.",
        "malayalam": "ഇന്ന് ഞാൻ ഒരു പുതിയ കാര്യം പഠിച്ചു.",
        "marathi": "आज मी एक नवीन गोष्ट शिकलो.",
        "gujarati": "આજે મેં એક નવી વસ્તુ શીખી.",
        "punjabi": "ਅੱਜ ਮੈਂ ਇੱਕ ਨਵੀਂ ਚੀਜ਼ ਸਿੱਖੀ।"
    }

    headers = {
        "Content-Type": "application/json"
    }

    response_times = []

    for lang, text in texts.items():

        payload = {
            "text": text,
            "ref_audio_path": "/home/aic_u2/Shubhankar/Livekit/github/annam-Livekit/tts/indicf5/IndicF5/prompts/PAN_F_HAPPY_00001.wav",
            "ref_text": "ਭਹੰਪੀ ਵਿੱਚ ਸਮਾਰਕਾਂ ਦੇ ਭਵਨ ਨਿਰਮਾਣ ਕਲਾ ਦੇ ਵੇਰਵੇ ਗੁੰਝਲਦਾਰ ਅਤੇ ਹੈਰਾਨ ਕਰਨ ਵਾਲੇ ਹਨ, ਜੋ ਮੈਨੂੰ ਖੁਸ਼ ਕਰਦੇ ਹਨ।"
        }

        print(f"Generating {lang}...")

        start_time = time.time()
        response = requests.post(url, json=payload, headers=headers)
        end_time = time.time()

        elapsed_time = round(end_time - start_time, 3)

        if response.status_code == 200:
            file_path = os.path.join(output_folder, f"{lang}.wav")
            with open(file_path, "wb") as f:
                f.write(response.content)
            print(f"{lang}.wav saved | {elapsed_time} sec")
        else:
            print(f"{lang} failed | Status: {response.status_code}")

        response_times.append([lang, elapsed_time])

    # Save CSV
    csv_file = "response_times.csv"
    with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Language", "Response_Time_sec"])
        writer.writerows(response_times)

    print(f"\nResponse times saved to {csv_file}")


if __name__ == "__main__":
    generate_tts_wav()
