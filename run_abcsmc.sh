#!/bin/bash
beta_values=(0.0001 0.001 0)
num_particles=1000
d=64
ed=32
enc_depth=6
dec_depth=4
mask=0.25

mkdir abc_logs

for beta in "${beta_values[@]}"; do
    # Define the directory path based on the beta value
    if (( $(bc -l <<< "$beta == 0") )); then
        dirpath="lotka_d${d}_ed${ed}_${enc_depth}_${dec_depth}_ae_mask_${mask}_conv_denoising"
    else
        dirpath="lotka_d${d}_ed${ed}_${enc_depth}_${dec_depth}_vae_mask_${mask}_beta_${beta}_conv_denoising"
    fi

    # Define the log file name
    log_file="${dirpath}_abc.log"

    # Run the ABC-SMC command in the background with nohup
    nohup python3 run_abcsmc.py \
        --path "$dirpath" \
        --tolerance_levels 0.15 0.125 0.1 0.07\
        --num_particles "$num_particles" \
        > "abc_logs/$log_file" 2>&1 &

    echo "Started ABC-SMC with beta=${beta}."
done
