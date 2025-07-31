#!/bin/bash
beta_values=(0 0.01 0.001 0.0001 0.00025)
d=64
eds=(64 32)
enc_depth=6
num_heads=8
dec_depths=(4)
decoder_num_heads=8
noise_factors=(0.0)
mask=0.15
num_particles=1000

for beta in "${beta_values[@]}"; do
    for ed in "${eds[@]}"; do
        for dec_depth in "${dec_depths[@]}"; do
            for noise in "${noise_factors[@]}"; do
                # Define the directory path based on the beta value
                if (( $(bc -l <<< "$beta == 0") )); then
                    dirpath="lotka_volterra_system/0.15/lotka_d${d}_ed${ed}_${enc_depth}_${num_heads}_${dec_depth}_${decoder_num_heads}_ae_mask_${mask}_noise_${noise}"
                else
                    dirpath="lotka_volterra_system/0.15/lotka_d${d}_ed${ed}_${enc_depth}_${num_heads}_${dec_depth}_${decoder_num_heads}_vae_mask_${mask}_beta_${beta}_noise_${noise}"
                fi

                # Define the log file name
                log_file="pairwise_abc_40.log"

                # Run the ABC-SMC command in the background with nohup
                nohup python3 run_abcpmc.py \
                    --config "$dirpath" \
                    --num_particles "$num_particles" \
                    > "$dirpath/$log_file" 2>&1 &

                echo "Started ABC-PMC with beta=${beta}."
            done
        done
    done
done
