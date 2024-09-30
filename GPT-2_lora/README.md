# Adapting GPT-2 using LoRA

This folder contains the implementation of [LoRA](https://github.com/microsoft/LoRA) in GPT-2 using the Python package `lora` and steps to replicate the results in our recent paper.


## Repository Overview

Our implementation is based on the fine-tuning code for GPT-2 in [Hugging Face](https://huggingface.co/).
There are several directories in this repo:
* [src/](src) contains the source code used for data processing, training, and decoding.
* [results/](results) contains the GPT-2 results files.
* [data/](data) contains the raw data we used in our experiments.
* [vocab/](vocab) contains the GPT-2 vocabulary files.


## Getting Started

 1. Clone our repo or the [LoRA](https://github.com/microsoft/LoRA) repo and install dependencies:
 ```
 
 bash download_pretrained_checkpoints.sh
 bash create_datasets.sh
 cd ./eval
 bash download_evalscript.sh
 cd ..
 ```

 ## Replicating Our Result on E2E

### 1. Train GPT-2 Medium with INNAprop using LoRA (see our paper for hyperparameters for GPT-2 Medium)
```
torchrun --standalone --nproc_per_node=1  src/gpt2_ft.py \
    --train_data ./data/e2e/train.jsonl \
    --valid_data ./data/e2e/valid.jsonl \
    --train_batch_size 8 \
    --grad_acc 1 \
    --valid_batch_size 4 \
    --seq_len 512 \
    --model_card gpt2.md \
    --init_checkpoint ./pretrained_checkpoints/gpt2-medium-pytorch_model.bin \
    --platform local \
    --clip 0.0 \
    --lr 0.0002 \
    --weight_decay 0.01 \
    --innaprop_alpha 0.1 \
    --innaprop_beta 0.9 \
    --scheduler linear \
    --warmup_step 500 \
    --max_epoch 5 \
    --save_interval 1000 \
    --lora_dim 4 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --label_smooth 0.1 \
    --work_dir ./INNAprop/trained_models/GPT2_M/e2e \
    --random_seed 5000 \
    --optimizer_name INNAprop

```

### 2. Train GPT-2 Medium with AdamW using LoRA (see our paper for hyperparameters for GPT-2 Medium)

```
torchrun --standalone --nproc_per_node=1 src/gpt2_ft.py \
    --train_data ./data/e2e/train.jsonl \
    --valid_data ./data/e2e/valid.jsonl \
    --train_batch_size 8 \
    --grad_acc 1 \
    --valid_batch_size 4 \
    --seq_len 512 \
    --model_card gpt2.md \
    --init_checkpoint ./pretrained_checkpoints/gpt2-medium-pytorch_model.bin \
    --platform local \
    --clip 0.0 \
    --lr 0.0002 \
    --weight_decay 0.01 \
    --correct_bias \
    --adam_beta2 0.999 \
    --scheduler linear \
    --warmup_step 500 \
    --max_epoch 5 \
    --save_interval 1000 \
    --lora_dim 4 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --label_smooth 0.1 \
    --work_dir ./AdamW/trained_models/GPT2_L/e2e \
    --random_seed 5000 \
    --optimizer_name AdamW
```

### 3. Generate outputs from the trained model using beam search:

```
torchrun --standalone --nproc_per_node=1 src/gpt2_beam.py \
    --data ./data/e2e/test.jsonl \
    --batch_size 1 \
    --seq_len 512 \
    --eval_len 64 \
    --model_card gpt2.md \
    --init_checkpoint ./INNAprop/trained_models/GPT2_M/e2e/model.26290.pt \
    --platform local \
    --lora_dim 4 \
    --lora_alpha 32 \
    --beam 10 \
    --length_penalty 0.8 \
    --no_repeat_ngram_size 4 \
    --repetition_penalty 1.0 \
    --eos_token_id 628 \
    --work_dir ./INNAprop/trained_models/GPT2_M/e2e \
    --output_file predict.26290.b10p08r4.jsonl
```

### 4. Decode outputs from step (2)
```
python src/gpt2_decode.py \
    --vocab ./vocab \
    --sample_file ./INNAprop/trained_models/GPT2_M/e2e/predict.26290.b10p08r4.jsonl \
    --input_file ./data/e2e/test_formatted.jsonl \
    --output_ref_file ./INNAprop/trained_models/GPT2_M/e2e/e2e_ref.txt \
    --output_pred_file ./INNAprop/trained_models/GPT2_M/e2e/e2e_pred.txt
```

### 5. Run evaluation on E2E test set

```
python eval/e2e/measure_scores.py INNAprop/trained_models/GPT2_M/e2e/e2e_ref.txt INNAprop/trained_models/GPT2_M/e2e/e2e_pred.txt -p
```

## Acknowledgement

The GPT-2 LoRA training code is is adapted from [LoRA](https://github.com/microsoft/LoRA). 