#!/bin/bash
beta_values=(0 0.01 0.1 1 3)
d=18
ed=9
enc_depth=3
dec_depth=1
mask=0.15

for beta in "${beta_values[@]}"; do
    # Define the directory path with the current beta value
    if (( $(bc -l <<< "$beta == 0") )); then
        dirpath="lotka_d${d}_ed${ed}_${enc_depth}_${dec_depth}_ae_mask_${mask}"
    else
        dirpath="lotka_d${d}_ed${ed}_${enc_depth}_${dec_depth}_vae_mask_${mask}_beta_${beta}"
    fi
    
    # Set the log file name
    log_file="${dirpath}_finetune.log"

    nohup python3 run_finetune.py --dirpath "$dirpath" --num_parameters 2 > "$log_file" 2>&1 &

    echo "Started fine-tuning with beta=${beta}."
done

# Wait for all background jobs to finish
# wait

echo "All training jobs started."

# End of file