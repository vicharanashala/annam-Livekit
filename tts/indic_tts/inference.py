"""
Minimal Indic TTS Inference Script
Uses Coqui TTS with FastPitch + HiFi-GAN for Indian languages

Supported Languages:
- Assamese, Bengali, Bodo, Gujarati, Hindi, Kannada, Malayalam,
- Manipuri, Marathi, Odia, Rajasthani, Tamil, Telugu
"""

import os
import argparse
from pathlib import Path
import torch
import numpy as np
from TTS.utils.synthesizer import Synthesizer


# Supported languages and their codes
SUPPORTED_LANGUAGES = {
    "as": "Assamese",
    "bn": "Bengali",
    "brx": "Bodo",
    "gu": "Gujarati",
    "hi": "Hindi",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mni": "Manipuri",
    "mr": "Marathi",
    "or": "Odia",
    "raj": "Rajasthani",
    "ta": "Tamil",
    "te": "Telugu",
}

# Available speakers (multi-speaker model)
SPEAKERS = ["female", "male"]


class IndicTTS:
    """Wrapper for Indic TTS inference using Coqui TTS framework."""

    def __init__(
        self,
        model_dir: str,
        lang: str = "hi",
        speaker: str = "female",
        use_cuda: bool = True,
    ):
        """
        Initialize Indic TTS.

        Args:
            model_dir: Path to the model directory containing lang subfolders
            lang: Language code (e.g., 'hi' for Hindi, 'ta' for Tamil)
            speaker: Speaker voice to use ('female' or 'male')
            use_cuda: Whether to use GPU for inference
        """
        self.model_dir = Path(model_dir)
        self.lang = lang
        self.speaker = speaker
        self.use_cuda = use_cuda and torch.cuda.is_available()

        if lang not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Language '{lang}' not supported. "
                f"Supported: {list(SUPPORTED_LANGUAGES.keys())}"
            )

        if speaker not in SPEAKERS:
            raise ValueError(
                f"Speaker '{speaker}' not supported. "
                f"Supported: {SPEAKERS}"
            )

        # Model paths (config expects models/v1/<lang>/... structure)
        lang_dir = self.model_dir / "v1" / lang
        self.model_path = lang_dir / "fastpitch" / "best_model.pth"
        self.config_path = lang_dir / "fastpitch" / "config.json"
        self.vocoder_path = lang_dir / "hifigan" / "best_model.pth"
        self.vocoder_config_path = lang_dir / "hifigan" / "config.json"

        # Validate paths
        self._validate_paths()

        # Initialize synthesizer
        self.synthesizer = Synthesizer(
            tts_checkpoint=str(self.model_path),
            tts_config_path=str(self.config_path),
            vocoder_checkpoint=str(self.vocoder_path),
            vocoder_config=str(self.vocoder_config_path),
            use_cuda=self.use_cuda,
        )

        print(f"Loaded Indic TTS model for {SUPPORTED_LANGUAGES[lang]}")
        print(f"Speaker: {speaker}")
        print(f"Using CUDA: {self.use_cuda}")

    def _validate_paths(self):
        """Validate that all required model files exist."""
        required_files = [
            self.model_path,
            self.config_path,
            self.vocoder_path,
            self.vocoder_config_path,
        ]
        for path in required_files:
            if not path.exists():
                raise FileNotFoundError(
                    f"Model file not found: {path}\n"
                    f"Please download models from: "
                    f"https://github.com/AI4Bharat/Indic-TTS/releases"
                )

    def synthesize(
        self,
        text: str,
        output_path: str = None,
        speaker: str = None,
    ) -> tuple:
        """
        Synthesize speech from text.

        Args:
            text: Input text in the target language
            output_path: Optional path to save the audio file
            speaker: Optional override speaker ('female' or 'male')

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        # Use provided speaker or default
        speaker_name = speaker or self.speaker

        # Generate audio with speaker name
        wav = self.synthesizer.tts(text, speaker_name=speaker_name)
        wav = np.array(wav)

        # Get sample rate from config
        sample_rate = self.synthesizer.tts_config.audio.sample_rate

        # Save if output path provided
        if output_path:
            self.synthesizer.save_wav(wav, output_path)
            print(f"Audio saved to: {output_path}")

        return wav, sample_rate


def main():
    parser = argparse.ArgumentParser(description="Indic TTS Inference")
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Input text to synthesize",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="hi",
        choices=list(SUPPORTED_LANGUAGES.keys()),
        help="Language code (default: hi for Hindi)",
    )
    parser.add_argument(
        "--speaker",
        type=str,
        default="female",
        choices=SPEAKERS,
        help="Speaker voice (default: female)",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./models",
        help="Path to model directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.wav",
        help="Output audio file path",
    )
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Disable CUDA (use CPU only)",
    )

    args = parser.parse_args()

    # Initialize TTS
    tts = IndicTTS(
        model_dir=args.model_dir,
        lang=args.lang,
        speaker=args.speaker,
        use_cuda=not args.no_cuda,
    )

    # Synthesize
    wav, sr = tts.synthesize(
        text=args.text,
        output_path=args.output,
    )

    print(f"Generated audio: {len(wav)} samples at {sr} Hz")


if __name__ == "__main__":
    main()
