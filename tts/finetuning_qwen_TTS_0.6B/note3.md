Resampled all 574 audio files from 44.1kHz to 24kHz using librosa and saved them in a new folder (test_p_24k) without modifying original files, as 24kHz is mandatory (assert sr == 24000).

Selected a single reference audio (10005878200211190857.wav, 7.14 sec) after checking durations, keeping it within the recommended 3–10 seconds range and used it for all samples to ensure speaker consistency.

Converted dataset from CSV to JSONL format, updated audio paths to resampled files, assigned the same reference audio, and created train_raw.jsonl with all 574 entries successfully.

Ran prepare_data.py to generate audio_codes, initially faced HuggingFace download issue due to XET protocol, fixed it using HF_HUB_DISABLE_XET=1, and successfully created train_with_codes.jsonl (7.3MB).

While running fine-tuning (sft_12hz.py), resolved multiple issues:

FlashAttention error → switched to attn_implementation="eager"

Tensor mismatch (2048 vs 1024) → added projection layer nn.Linear(2048, 1024)

Model path error → used correct HuggingFace cached path

Encountered limitation of no sudo access (SoX installation) on VM.