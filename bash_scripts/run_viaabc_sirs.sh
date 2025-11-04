#!/bin/bash
beta_values=(0 0.1 0.01 0.001)
d=128
eds=(64)
enc_depth=6
num_heads=8
dec_depths=(4)
decoder_num_heads=8
mask=0.15


num_particles=1000
idx=0
k=5


for beta in "${beta_values[@]}"; do
    if (( $(bc -l <<< "$beta == 0") )); then
        dirpath="Sirs/spatialSIR3D_d${d}_ed64_${enc_depth}_${num_heads}_4_${decoder_num_heads}_ae_mask_${mask}"
    else
        dirpath="Sirs/spatialSIR3D_d${d}_ed64_${enc_depth}_${num_heads}_4_${decoder_num_heads}_vae_mask_${mask}_beta_${beta}"
    fi

    # Define the log file name
    log_file="pairwise_${idx}.log"

    # Run the ABC-SMC command in the background with nohup
    nohup python3 run_scripts/run_viaabc.py \
        --folder_name "$dirpath" \
        --num_particles "$num_particles" \
        --idx "$idx" \
        --k "$k" \
        > "$dirpath/$log_file" 2>&1 &

    echo "Started viaABC with beta=${beta}."
done
