1.XTTSv2

  File "/home/deepthi/miniconda3/envs/venv_xttsv2/lib/python3.10/site-packages/trainer/trainer.py", line 2098, in get_criterion
    criterion = model.get_criterion()
  File "/home/deepthi/miniconda3/envs/venv_xttsv2/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1688, in __getattr__
    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
AttributeError: 'Xtts' object has no attribute 'get_criterion'

the key point:

“not supported xtts finetuning”

2. Mistral tts=Voxtral
 not supporting finetuned model
