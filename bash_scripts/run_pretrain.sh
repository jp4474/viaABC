#!/bin/bash
beta_values=(0 0.01 0.001 0.0001)
d=64
ed=48
enc_depth=2
dec_depth=2
num_heads=4
decoder_num_heads=4
noise_factor=0.0
mask=0.15

for beta in "${beta_values[@]}"; do
    # Define the directory path with the current beta value
    if (( $(bc -l <<< "$beta == 0") )); then
        dirpath="CAR_d${d}_ed${ed}_${enc_depth}_${num_heads}_${dec_depth}_${decoder_num_heads}_ae_mask_${mask}_noise_${noise_factor}"
    else
        dirpath="CAR_d${d}_ed${ed}_${enc_depth}_${num_heads}_${dec_depth}_${decoder_num_heads}_vae_mask_${mask}_beta_${beta}_noise_${noise_factor}"
    fi
    
    # Set the log file name
    log_file="${dirpath}.log"

    if (( $(bc -l <<< "$beta != 0") )); then
        # Run the command in the background with nohup
        nohup python3 run_scripts/run_pretrain.py \
            --dirpath "$dirpath" \
            --type vae \
            --beta "$beta" \
            --embed_dim "$d" \
            --num_heads "$num_heads" \
            --depth "$enc_depth" \
            --decoder_depth "$dec_depth" \
            --decoder_embed_dim "$ed" \
            --decoder_num_heads "$decoder_num_heads" \
            --mask_ratio "$mask" \
            --noise_factor "$noise_factor" \
            > "$log_file" 2>&1 &
    else 
        nohup python3 run_scripts/run_pretrain.py \
            --dirpath "$dirpath" \
            --type vanilla \
            --beta "$beta" \
            --embed_dim "$d" \
            --num_heads "$num_heads" \
            --depth "$enc_depth" \
            --decoder_depth "$dec_depth" \
            --decoder_embed_dim "$ed" \
            --decoder_num_heads "$decoder_num_heads" \
            --mask_ratio "$mask" \
            --noise_factor "$noise_factor" \
            > "$log_file" 2>&1 &
    fi

    echo "Started training with beta=${beta}."
done

# End of file
