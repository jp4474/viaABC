#!/bin/bash

nohup python3 run_pretrain.py --dirpath mzb_d64_ed64_6_4_b0.1 --type vae --beta 0.1 --embed_dim 64 --num_heads 8 --depth 6 --decoder_depth 4 --decoder_embed_dim 64 --decoder_num_heads 8 --mask_ratio 0.5 > mzb_d64_ed64_6_4_b0.1.log 2>&1 &
nohup python3 run_pretrain.py --dirpath mzb_d64_ed64_6_4_b0.01 --type vae --beta 0.01 --embed_dim 64 --num_heads 8 --depth 6 --decoder_depth 4 --decoder_embed_dim 64 --decoder_num_heads 8 --mask_ratio 0.5 > mzb_d64_ed64_6_4_b0.01.log 2>&1 &
nohup python3 run_pretrain.py --dirpath mzb_d64_ed64_6_4_b0.001 --type vae --beta 0.001 --embed_dim 64 --num_heads 8 --depth 6 --decoder_depth 4 --decoder_embed_dim 64 --decoder_num_heads 8 --mask_ratio 0.5 > mzb_d64_ed64_6_4_b0.001.log 2>&1 &
nohup python3 run_pretrain.py --dirpath mzb_d64_ed64_6_4_b1 --type vae --beta 1 --embed_dim 64 --num_heads 8 --depth 6 --decoder_depth 4 --decoder_embed_dim 64 --decoder_num_heads 8 --mask_ratio 0.5 > mzb_d64_ed64_6_4_b1.log 2>&1 &

# End of file