"""
Download Indic TTS model weights from GitHub releases.

Model weights for all 13 languages are available at:
https://github.com/AI4Bharat/Indic-TTS/releases/tag/v1-checkpoints-release
"""

import os
import argparse
import subprocess
from pathlib import Path

# Model download URLs (from AI4Bharat releases)
# Note: These need to be updated if the URLs change
MODEL_URLS = {
    "hi": "https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release/hi.zip",
    "ta": "https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release/ta.zip",
    "te": "https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release/te.zip",
    "bn": "https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release/bn.zip",
    "kn": "https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release/kn.zip",
    "ml": "https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release/ml.zip",
    "mr": "https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release/mr.zip",
    "gu": "https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release/gu.zip",
    "or": "https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release/or.zip",
    "as": "https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release/as.zip",
    "brx": "https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release/brx.zip",
    "mni": "https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release/mni.zip",
    "raj": "https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release/raj.zip",
}


def download_model(lang: str, models_dir: Path):
    """Download and extract model for a specific language."""
    if lang not in MODEL_URLS:
        print(f"Unknown language: {lang}")
        return False

    url = MODEL_URLS[lang]
    zip_path = models_dir / f"{lang}.zip"
    lang_dir = models_dir / lang

    if lang_dir.exists():
        print(f"Model for '{lang}' already exists at {lang_dir}")
        return True

    print(f"Downloading {lang} model from {url}...")

    try:
        # Download using wget or curl
        subprocess.run(
            ["wget", "-O", str(zip_path), url],
            check=True,
        )

        # Extract
        print(f"Extracting {lang} model...")
        subprocess.run(
            ["unzip", "-o", str(zip_path), "-d", str(models_dir)],
            check=True,
        )

        # Cleanup zip
        zip_path.unlink()
        print(f"Successfully downloaded and extracted {lang} model")
        return True

    except subprocess.CalledProcessError as e:
        print(f"Error downloading {lang}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download Indic TTS models")
    parser.add_argument(
        "--lang",
        type=str,
        default="hi",
        choices=list(MODEL_URLS.keys()) + ["all"],
        help="Language to download (use 'all' for all languages)",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="./models",
        help="Directory to save models",
    )

    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    if args.lang == "all":
        for lang in MODEL_URLS:
            download_model(lang, models_dir)
    else:
        download_model(args.lang, models_dir)


if __name__ == "__main__":
    main()
