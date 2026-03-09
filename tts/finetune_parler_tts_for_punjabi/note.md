To train or fine-tune Parler-TTS, the dataset must contain three key elements: speech audio, the corresponding transcription, and a text description of speech characteristics (e.g., speaker style or noise level). During training, audio is first converted into discrete audio tokens using the DAC audio encoder, and the model is trained to predict these tokens from text inputs.
Training can be performed either from scratch or by fine-tuning a pre-trained checkpoint such as parler-tts-mini-v1, with fine-tuning being more practical when compute resources are limited. The training process is managed by the run_parler_tts_training.py script and configured through command-line arguments or a JSON configuration file.
Configuration settings define the model to use, datasets to load, column mappings for audio and text, and key hyperparameters such as batch size, learning rate, and number of epochs. The framework also supports combining multiple datasets and saving intermediate processed data (audio tokens) to avoid recomputation.





Fixed the multi-GPU timeout error by forcing training to run on single GPU using --num_processes=1

Installed all missing libraries — wandb, evaluate, jiwer, torchmetrics and torchcodec

Successfully completed fine-tuning on 10 audio samples and model saved at parler_output/

Wrote and ran inference script to test the fine-tuned model

Faced CUDA out of memory error on GPU inference due to other users occupying the shared GPU so switched to CPU inference

Successfully generated audio output from the fine-tuned model

Full fine-tuning and inference pipeline is working correctly end to end