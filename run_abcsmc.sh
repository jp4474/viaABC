#!/bin/bash
beta_values=(0 0.001 0.01 0.1 1)
num_particles=1000
d=128
ed=64
enc_depth=8
dec_depth=4
mask=0.5

for beta in "${beta_values[@]}"; do
    # Define the directory path based on the beta value
    if (( $(bc -l <<< "$beta == 0") )); then
        dirpath="lotka_d${d}_ed${ed}_${enc_depth}_${dec_depth}_ae_mask_${mask}_denoising"
    else
        dirpath="lotka_d${d}_ed${ed}_${enc_depth}_${dec_depth}_vae_mask_${mask}_beta_${beta}_denoising"
    fi

    # Define the log file name
    log_file="${dirpath}_abc_normal.log"

    # Run the ABC-SMC command in the background with nohup
    nohup python3 run_abcsmc.py \
        --path "$dirpath" \
        --tolerance_levels 0.2 0.18 0.16 0.14 0.12\
        --num_particles "$num_particles" \
        > "$log_file" 2>&1 &

    echo "Started ABC-SMC with beta=${beta}."
done

# nohup python3 run_abcsmc.py --path lotka_d18_ed9_3_1_ae_mask_0.15 --tolerance_levels 0.3 --num_particles 1000 > lotka_d32_ed32_abc_ae_test.log 2>&1 &
# nohup python3 run_abcsmc.py --path "lotka_d18_ed9_3_1_vae_mask_0.15_beta_0.01" --tolerance_levels 0.2 0.15 0.1 0.05 --num_particles 1000 > lotka_d32_ed32_abc_vae_0.01.log 2>&1 &
# nohup python3 run_abcsmc.py --path "lotka_d18_ed9_3_1_vae_mask_0.15_beta_1" --tolerance_levels 0.2 0.15 0.1 0.05 --num_particles 1000 > lotka_d32_ed32_abc_vae_1.log 2>&1 &
# nohup python3 run_abcsmc.py --path "lotka_d18_ed9_3_1_vae_mask_0.15_beta_4" --tolerance_levels 0.2 0.15 0.1 0.05 --num_particles 1000 > lotka_d32_ed32_abc_vae_4.log 2>&1 &
# nohup python3 run_abcsmc.py --path "lotka_d18_ed9_3_1_vae_mask_0.15_beta_100" --tolerance_levels 0.2 0.15 0.1 0.05 --num_particles 1000 > lotka_d32_ed32_abc_vae_100.log 2>&1 &

# nohup python3 run_abcsmc.py --path "lotka_d128_ed64_8_4_vae_mask_0.15_beta_1" --tolerance_levels 0.2 0.15 0.1 0.05 0.01 --num_particles 1000 --finetune > lotka_d128_ed64_abc_finetune.log 2>&1 &
# nohup python3 run_abcsmc.py --path "lotka_d512_ed256_6_4_vae_mask_0.15_beta_1" --tolerance_levels 0.2 0.15 0.1 0.05 0.01 --num_particles 1000 --finetune > lotka_d512_ed256_abc_finetune.log 2>&1 &

# End of file