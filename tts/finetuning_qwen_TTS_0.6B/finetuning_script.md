CUDA_VISIBLE_DEVICES=1 python sft_12hz.py \
--init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
--output_model_path /home/koustav/fine_tuning/Qwen3-TTS/finetuning/qwen_tts_1.7b_output_v2 \
--train_jsonl train_with_codes.jsonl \
--batch_size 2 \
--lr 2e-6 \
--num_epochs 20 \
--speaker_name punjabi_speaker