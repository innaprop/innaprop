## Reproduce GPT-2 Results

Prepare the [OpenWebText](https://huggingface.co/datasets/openwebtext) data following [nanoGPT](https://github.com/karpathy/nanoGPT/):
```
$ python data/openwebtext/prepare.py
```
Start pre-training GPT2 Small (125M):

To start pre-training GPT-2 Small (125M), use the following command if you have access to a machine with 8 A100 (80GB) GPUs:
```
$ torchrun --standalone --nproc_per_node=8 \
      train.py \
      config/train_gpt2_small_innaprop.py \
      --batch_size=12 \
      --gradient_accumulation_steps=8
```

To reproduce the AdamW baseline as described in [nanoGPT](https://github.com/karpathy/nanoGPT/), use the following command:
```
$ torchrun --standalone --nproc_per_node=8 \
      train.py \
      config/train_gpt2_small_adam.py \
      --batch_size=12 \
      --gradient_accumulation_steps=8
```

If you are using a different hardware setup, you can adjust the ``nproc_per_node```, ```batch_size```, and ```gradient_accumulation_steps``` parameters. Ensure that the product of these values equals 480 for consistent training behavior.

## Acknowledgement

The GPT-2 training code is is adapted from [nanoGPT](https://github.com/karpathy/nanoGPT/). 