#!/bin/bash
beta_values=(0 0.01 0.1 1 3)
d=64
ed=32
enc_depth=4
dec_depth=2
mask=0.15

for beta in "${beta_values[@]}"; do
    # Use bc for the floating point comparison and capture its exit value
    if [ "$(echo "$beta == 0" | bc -l)" -eq 1 ]; then
        dirpath="lotka_d${d}_ed${ed}_${enc_depth}_${dec_depth}_ae_mask_${mask}_multitasks"
        type_arg="vanilla"
    else
        dirpath="lotka_d${d}_ed${ed}_${enc_depth}_${dec_depth}_vae_mask_${mask}_beta_${beta}_multitasks"
        type_arg="vae"
    fi

    log_file="${dirpath}.log"

    nohup python3 run_pretrain.py \
        --dirpath "$dirpath" \
        --type "$type_arg" \
        --beta "$beta" \
        --embed_dim "$d" \
        --num_heads 8 \
        --depth "$enc_depth" \
        --decoder_depth "$dec_depth" \
        --decoder_embed_dim "$ed" \
        --decoder_num_heads 8 \
        --mask_ratio "$mask" \
        --multi_tasks \
        > "$log_file" 2>&1 &

    echo "Started training with beta=${beta}."
done

# Wait for all background jobs to finish

# d=64
# ed=32
# enc_depth=4
# dec_depth=2
# mask=0.15

# d=128
# ed=64
# enc_depth=8
# dec_depth=4

# d=512
# ed=256
# enc_depth=6
# dec_depth=4

# End of file