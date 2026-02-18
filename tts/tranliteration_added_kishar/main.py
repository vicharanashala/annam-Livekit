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

# result = transliterate_en_to_indic(
#     text="Mujhe pytorch bahat accha lagta hai. Suna he ki weight update ke liye backpropagation use hota hai.",
#     target_languages=["hi", "bn"]
# )

# print(json.dumps(result, indent=2, ensure_ascii=False))
sentences = [
    "Mujhe PyTorch bahut accha lagta hai, kyuki isme dynamic computation graph hota hai.",
    "Suna hai ki neural network me weight update ke liye backpropagation use hota hai.",
    "Loss function minimize karne ke liye gradient descent algorithm lagta hai.",
    "Optimizer jaise Adam aur SGD learning rate ko control karte hain.",
    "Har layer me activation function jaise ReLU ya sigmoid non-linearity lata hai.",
    "Forward pass me input tensor layers se pass hota hai aur output logits milte hain.",
    "Backward pass me autograd gradients calculate karta hai automatically.",
    "Batch normalization training ko stable banata hai aur vanishing gradient kam karta hai.",
    "Dropout regularization overfitting ko reduce karta hai.",
    "CNN models convolution aur pooling layers se spatial features extract karte hain.",
    "RNN aur LSTM sequence data me temporal dependencies pakadte hain.",
    "Transformer models attention mechanism se context samajhte hain.",
    "GPU acceleration CUDA cores ki wajah se training fast ho jati hai.",
    "Dataset ko DataLoader batches me divide karta hai memory optimize karne ke liye.",
    "End me model inference karta hai aur softmax probabilities output deta hai."
]

import json

output_lines = []

for sentence in sentences:
    result = transliterate_en_to_indic(
        text=sentence,
        target_languages=["hi", "bn"]
    )

    # Print nicely
    print("Original:", sentence)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("-" * 50)

    # Save formatted text
    block = f"Original: {sentence}\n{json.dumps(result, ensure_ascii=False, indent=2)}\n"
    output_lines.append(block)

# Save to txt file
with open("transliteration_output.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output_lines))

print("Saved to transliteration_output.txt")
