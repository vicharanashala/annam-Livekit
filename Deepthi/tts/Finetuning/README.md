# 1.XTTSv2

  File "/home/deepthi/miniconda3/envs/venv_xttsv2/lib/python3.10/site-packages/trainer/trainer.py", line 2098, in get_criterion
    criterion = model.get_criterion()
  File "/home/deepthi/miniconda3/envs/venv_xttsv2/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1688, in __getattr__
    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
AttributeError: 'Xtts' object has no attribute 'get_criterion'

the key point:

“not supported xtts finetuning”

# 2. Mistral tts=Voxtral
 not supporting finetuned model

# 3. Cosyvoice 2
   Tried finetuninfg the latest version - installed all the packages  but many of the files are not suitable 
# 4. Sesame -csm 1b
    Tried fine tuning using small subset of Hindi dataset(train-3000 and val - 600) epoch 50
   testing- produced uncleared voice.

   # Steps
   https://github.com/knottwill/sesame-finetune

# Convert train.txt and val.txt into .json format
Deepthi/TTS/Finetuning/IndicSynth_processed/Hindi/train.json
Deepthi/TTS/Finetuning/IndicSynth_processed/Hindi/val.json

echo "WANDB_API_KEY=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" > .env
echo "CSM_REPO_PATH=/home/deepthi/Deepthi/TTS/Finetuning/Sesame/csm" >> .env
echo "AUDIO_NUM_CODEBOOKS=32"



python pretokenize.py \
  --train_data /home/deepthi/Deepthi/TTS/Finetuning/IndicSynth_processed/Hindi/train.json \
  --val_data /home/deepthi/Deepthi/TTS/Finetuning/IndicSynth_processed/Hindi/val.json \
  --output /home/deepthi/Deepthi/TTS/Finetuning/IndicSynth_processed/Hindi/tokenized_data.hdf5

# to start training using pretrained weights
python train.py \
  --data /home/deepthi/Deepthi/TTS/Finetuning/IndicSynth_processed/Hindi/tokenized_data.hdf5 \
  --config ./configs/finetune_param_defaults.yaml \
  --n_epochs 25 \
  --gen_every 500 \
  --gen_sentence "यह एक परीक्षण वाक्य है"

# to start training from scratch

python train.py \
  --data /home/deepthi/Deepthi/TTS/Finetuning/IndicSynth_processed/Hindi/tokenized_data.hdf5 \
  --config ./configs/finetune_param_defaults.yaml \
  --train_from_scratch \
  --n_epochs 1 \
  --gen_every 100 \
  --gen_sentence "यह एक परीक्षण वाक्य है"


test.py path
Deepthi/TTS/Finetuning/Sesame/sesame-finetune/test.py

# ................... using Subset ................

# For creating subset
shuf train.txt | head -n 3000 > train_subset.txt
shuf val.txt | head -n 600 > val_subset.txt

Deepthi/TTS/Finetuning/IndicSynth_processed/Hindi/train_subset.txt

for adding root
sed -i 's|^|/home/deepthi/|' train_subset.txt



(venv_sesame) deepthi@b389743eceac:~/Deepthi/TTS/Finetuning/Sesame/sesame-finetune$

Golden rule (remember this)

Before EVERY training / pretokenization, run:

export $(cat .env | xargs)


python pretokenize.py \
  --train_data /home/deepthi/Deepthi/TTS/Finetuning/IndicSynth_processed/Hindi/train_subset.json \
  --val_data /home/deepthi/Deepthi/TTS/Finetuning/IndicSynth_processed/Hindi/val_subset.json \
  --output /home/deepthi/Deepthi/TTS/Finetuning/IndicSynth_processed/Hindi/tokenized_subset.hdf5


# if no pretrained model 
python train.py \
  --data /home/deepthi/Deepthi/TTS/Finetuning/IndicSynth_processed/Hindi/tokenized_subset_fixed.hdf5 \
  --config ./configs/finetune_param_defaults.yaml \
  --model_name_or_checkpoint_path sesame/csm-1b \
  --n_epochs 50 \
  --gen_every 500 \
  --gen_sentence "यह एक परीक्षण वाक्य है"

# training from scratch
....................................
export AUDIO_NUM_CODEBOOKS=32
export WANDB_MODE=offline

python train.py \
  --data /home/deepthi/Deepthi/TTS/Finetuning/IndicSynth_processed/Hindi/tokenized_subset.hdf5 \
  --config ./configs/finetune_param_defaults.yaml \
  --train_from_scratch \
  --n_epochs 50 \
  --gen_every 500 \
  --gen_sentence "यह एक परीक्षण वाक्य है"
........................................................
# By using pretrained weights
python train.py \
  --data /home/deepthi/Deepthi/TTS/Finetuning/IndicSynth_processed/Hindi/tokenized_subset.hdf5 \
  --config ./configs/finetune_param_defaults.yaml \
  --n_epochs 25 \
  --gen_every 500 \
  --gen_sentences "यह एक परीक्षण वाक्य है"
OR...............................
python train.py \
  --data /home/deepthi/Deepthi/TTS/Finetuning/IndicSynth_processed/Hindi/tokenized_subset.hdf5 \
  --config ./configs/finetune_param_defaults.yaml \
  --model_name_or_checkpoint_path ./exp/model_18749.pt \
  --n_epochs 50 \
  --gen_every 500 \
  --gen_sentences "यह एक परीक्षण वाक्य है"

Do NOT train from scratch
👉 Use pretrained model

# for checking

https://wandb.ai/deepthiajith-iit-ropar-tif
https://wandb.ai/deepthiajith-iit-ropar-tif/csm-finetuning/runs/s3w9tu3w?nw=nwuserdeepthiajith14

# 5 svara -tts 
kenpath/svara-tts-v1
https://huggingface.co/kenpath/svara-tts-v1?utm_source=chatgpt.com
At a Glance
Languages (19): Hindi, Bengali, Marathi, Telugu, Kannada, Bhojpuri, Magahi, Chhattisgarhi, Maithili, Assamese, Bodo, Dogri, Gujarati, Malayalam, Punjabi, Tamil, Nepali, Sanskrit, Indian English.
Expressivity: End-of-utterance style tags; natural prosody; code-switch aware.
Latency & Deployment: Works well with GGUF exports; suitable for edge/CPU scenarios.
Adaptability: LoRA-friendly for quick speaker/domain specialization.

# Steps
# UPDATE train.yaml (important for GPU)
batch_size: 32   # (start safe, can increase later)
num_workers: 2   # try 2, if shm error → reduce to 0
epochs=70

# RUN TRAINING ON CUDA
python train.py --device=cuda:0 \
--data_folder=/home/deepthi/Deepthi/TTS/Finetuning/svara/tts_data \
hparams/train.yaml



