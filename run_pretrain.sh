#!/bin/bash

# nohup python3 run_pretrain.py --dirpath vae_32_2_1_1 --type vae --beta 1 --embed_dim 32 --decoder_embed_dim 32 > vae_64_2_1_1.log 2>&1 &
# nohup python3 run_pretrain.py --dirpath vae_32_2_1_2 --type vae --beta 2 --embed_dim 32 --decoder_embed_dim 32 > vae_64_2_1_2.log 2>&1 &
# nohup python3 run_pretrain.py --dirpath vae_32_2_1_3 --type vae --beta 3 --embed_dim 32 --decoder_embed_dim 32 > vae_64_2_1_3.log 2>&1 &

nohup python3 run_pretrain.py --dirpath vae_32_2_1_1_nd --type vae --beta 1 --embed_dim 32 --decoder_embed_dim 32 --dropout 0 > vae_64_2_1_1.log 2>&1 &
nohup python3 run_pretrain.py --dirpath vae_32_2_1_2_nd --type vae --beta 2 --embed_dim 32 --decoder_embed_dim 32 --dropout 0 > vae_64_2_1_2.log 2>&1 &
nohup python3 run_pretrain.py --dirpath vae_32_2_1_3_nd --type vae --beta 3 --embed_dim 32 --decoder_embed_dim 32 --dropout 0 > vae_64_2_1_3.log 2>&1 &

# End of file