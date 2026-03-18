Qwen3-TTS is a codec-based TTS model that generates audio using discrete audio tokens instead of raw waveforms.


It uses a dual-channel architecture (text + audio codec), which is different from traditional TTS models like Parler.


The model supports multilingual speech generation and currently allows single-speaker fine-tuning.


Dataset format is JSONL, where each sample contains audio, text, and ref_audio.


Reference audio is mandatory and should ideally be the same across all samples for better speaker consistency.


All audio must be resampled to 24kHz, which is a strict requirement for this model.


A separate preprocessing step (prepare_data.py) is required to convert audio into audio_codes before training.


Audio codes are generated using a tokenizer and represent speech as discrete tokens, making training a sequence prediction task.


Training is done using Supervised Fine-Tuning (SFT) with the sft_12hz.py script.


Important training parameters include learning rate, batch size, epochs, and speaker name.


The 0.6B model can use a slightly higher learning rate (~2e-5) compared to the 1.7B model.


Checkpoints are saved after every epoch, allowing evaluation at different stages.


The dataset pipeline internally uses mel spectrograms (128 bins) for reference audio processing.


The model uses multiple masks and special tokens to properly align text and audio during training.


Compared to Parler TTS, the pipeline is more complex but provides better voice cloning and flexibility.


Key implementation steps include CSV → JSONL conversion, audio resampling, reference audio selection, data preparation, and fine-tuning.


Main challenges include data preprocessing, maintaining audio quality, and evaluating generated speech.