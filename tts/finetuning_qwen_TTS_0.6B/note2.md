Started working on fine-tuning Qwen TTS 0.6B and identified dataset as the primary requirement.

Explored multiple datasets including LibriTTS / LibriTTS-R (585 hours, 24kHz), GigaSpeech (10,000 hours), and other TTS datasets like Kituba and Suundi.

Faced issues with dataset accessibility and usability, especially with downloading and format compatibility.

Decided to use the existing Punjabi dataset (used in Parler TTS) due to dataset challenges.

Analyzed differences between existing dataset and Qwen TTS required format (JSONL + ref_audio).

Identified key preprocessing steps:

Resampling audio from 44.1kHz → 24kHz

Selecting a single high-quality reference audio

Converting CSV → JSONL format

Cloned the Qwen3-TTS repository into the fine-tuning environment.

Verified all required scripts (prepare_data.py, sft_12hz.py, dataset.py, README).

Created a separate virtual environment (qwen_env) to avoid conflicts with Parler TTS setup.

Upgraded pip and installed qwen-tts and all required dependencies (torch, transformers, librosa, etc.).