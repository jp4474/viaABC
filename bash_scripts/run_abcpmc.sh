#!/bin/bash
beta_values=(0 0.01 0.001 0.0001)
d=64
eds=(32)
enc_depth=2
num_heads=4
dec_depths=(2)
decoder_num_heads=4
noise_factors=(0.0)
mask=0.15
num_particles=1000
idx=0

for beta in "${beta_values[@]}"; do
    for ed in "${eds[@]}"; do
        for dec_depth in "${dec_depths[@]}"; do
            for noise in "${noise_factors[@]}"; do
                # Define the directory path based on the beta value
                if (( $(bc -l <<< "$beta == 0") )); then
                    dirpath="CAR_d${d}_ed${ed}_${enc_depth}_${num_heads}_${dec_depth}_${decoder_num_heads}_ae_mask_${mask}_noise_${noise}_Log"
                else
                    dirpath="CAR_d${d}_ed${ed}_${enc_depth}_${num_heads}_${dec_depth}_${decoder_num_heads}_vae_mask_${mask}_beta_${beta}_noise_${noise}_Log"
                fi

                # dirpath="/home/jp4474/latent-abc-smc/notebooks_bcell/"
                # Define the log file name
                log_file="pairwise_${idx}.log"

                # Run the ABC-SMC command in the background with nohup
                nohup python3 run_scripts/run_abcpmc.py \
                    --folder_name "$dirpath" \
                    --num_particles "$num_particles" \
                    --idx "$idx" \
                    > "$dirpath/$log_file" 2>&1 &

                echo "Started viaABC with beta=${beta}."
            done
        done
    done
done
