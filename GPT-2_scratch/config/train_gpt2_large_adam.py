wandb_log = True
wandb_project = 'innaprop'
wandb_run_name=f'gpt2-large-adam-100k'


# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 8

n_layer = 36
n_head = 20
n_embd = 1280
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False

max_iters = 100000
lr_decay_iters = 100000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# optimizer
optimizer_name = 'adamw'
learning_rate = 2e-4 # max learning rate
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
min_lr = 1e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

compile = False
dtype = 'float16'

out_dir = f'out_large-adam-100k'
