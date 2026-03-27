CUDA_VISIBLE_DEVICES=1 python sft_12hz.py \
--train_jsonl train_with_codes.jsonl \
--output_model_path qwen_tts_1.7b_output \
--batch_size 2 \
--lr 2e-5 \
--num_epochs 20 \
--speaker_name punjabi_speaker
