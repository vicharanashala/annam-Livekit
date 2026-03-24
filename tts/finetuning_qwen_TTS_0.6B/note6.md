Identified issue in Qwen 0.6B, started exploring Qwen TTS 1.7B.
Fixed sft_12hz.py to dynamically detect local model path.
Started fine-tuning; fixed path error after 1st epoch and restarted training.
Completed fine-tuning (10 epochs) with loss reduced 13.20 → 5.38.
Generated audio → better quality than 0.6B, good tone, pronunciation, and consistency.
Tested emotion, long text, and out-of-domain (English) → mostly good, minor issues.
