import requests
import json

BASE_URL = "https://mesne-unlicentiously-allie.ngrok-free.dev/xlit"

SUPPORTED_LANGS = [
    "as", "bn", "brx", "en", "gom", "gu", "hi", "kn", "ks",
    "mai", "ml", "mni", "mr", "ne", "or", "pa", "sa",
    "sd", "si", "ta", "te", "ur"
]

def transliterate_en_to_indic(text, target_languages, beam_width=4):
    """
    Calls En-to-Indic transliteration API
    """

    # Validate languages
    invalid = [lang for lang in target_languages if lang not in SUPPORTED_LANGS]
    if invalid:
        raise ValueError(f"Unsupported language(s): {invalid}")

    payload = {
        "text": text,
        "target_languages": target_languages,
        "beam_width": beam_width
    }

    try:
        response = requests.post(
            f"{BASE_URL}/transliterate/en-to-indic",
            json=payload
        )

        response.raise_for_status()
        return response.json()

    except Exception as e:
        return {"success": False, "error": str(e)}


 
 

result = transliterate_en_to_indic(
    text="Ph No",
    target_languages=["ta"]
)

print(json.dumps(result, indent=2, ensure_ascii=False))