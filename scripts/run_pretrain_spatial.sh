#!/bin/bash
beta_values=(0.01)
mask=0.15
d=128
ed=64
enc_depth=6
dec_depth=4
num_heads=8
decoder_num_heads=8
patch_size=8
t_patch_size=3

for beta in "${beta_values[@]}"; do
    # Define the directory path with the current beta value
    if (( $(bc -l <<< "$beta == 0") )); then
        dirpath="spatialSIR3D_d${d}_ed${ed}_${enc_depth}_${num_heads}_${dec_depth}_${decoder_num_heads}_ae_mask_${mask}"
    else
        dirpath="spatialSIR3D_d${d}_ed${ed}_${enc_depth}_${num_heads}_${dec_depth}_${decoder_num_heads}_vae_mask_${mask}_beta_${beta}"
    fi
    
    # Set the log file name
    log_file="${dirpath}.log"

    if (( $(bc -l <<< "$beta != 0") )); then
        # Run the command in the background with nohup
        nohup python3 run_pretrain_spatial.py \
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
            --patch_size "$patch_size" \
            --t_patch_size "$t_patch_size" \
            > "$log_file" 2>&1 &
    else 
        nohup python3 run_pretrain_spatial.py \
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
            --patch_size "$patch_size"  \
            --t_patch_size "$t_patch_size" \
            > "$log_file" 2>&1 &
    fi

    echo "Started training with beta=${beta}."
done

# End of file

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
