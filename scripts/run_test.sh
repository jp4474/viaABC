beta_values=(0 0.01 0.001 0.1)
d=128
ed=64
enc_depth=6
num_heads=8
dec_depth=4
decoder_num_heads=8
mask=0.15
metric="pairwise_cosine"

# Loop over each beta
for beta in "${beta_values[@]}"; do
    # Determine directory name based on beta
    if (( $(bc -l <<< "$beta == 0") )); then
        dirpath="spatialSIR3D_d${d}_ed${ed}_${enc_depth}_${num_heads}_${dec_depth}_${decoder_num_heads}_ae_mask_${mask}"
    else
        dirpath="spatialSIR3D_d${d}_ed${ed}_${enc_depth}_${num_heads}_${dec_depth}_${decoder_num_heads}_vae_mask_${mask}_beta_${beta}"
    fi

    full_path="/home/jp4474/latent-abc-smc/${dirpath}"
    echo "Running inference for beta=$beta in folder $full_path"
    
    nohup python3 test.py \
        --folder "$full_path" \
        --metric "$metric" > "${full_path}/${metric}_4.log" 2>&1 &
done