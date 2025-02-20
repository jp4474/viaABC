# latent-abc-smc

LatentABCSMC implements a Sequential Monte Carlo (SMC) algorithm for Approximate Bayesian Computation (ABC). This Python class is designed to estimate posterior distributions of model parameters when direct likelihood calculations are infeasible due to computational constraints or the absence of explicit likelihood functions for the data. The algorithm extends the standard ABC-SMC framework by leveraging a pre-trained Variational Autoencoder (VAE) to encode data into a latent representation. This approach enables the computation of distances in the latent space, offering a more effective alternative to traditional summary statistics or conventional distance metrics.


# How to train?
Configure your simulation in `latent_abc_smc.py` 

Configure `generate_training_data.py` and call the configured class from above. 

Execute

```bash
  python3 generate_training_data.py --train_sizes [500000, 50000, 50000]
```

to generate training, validation, and testing data.

Configure `run_pretrain.py` and execute

Example:

```bash
  python3 run_pretrain.py --dirpath lotka_d18_ed9_3_1_vae_mask_0.75_beta_1 \
--type vae --beta 1 --embed_dim 18 --num_heads 6 --depth 3 \
--decoder_depth 1 --decoder_embed_dim 9 --decoder_num_heads 3 \
--mask_ratio 0.15 > lotka_d18_ed9_3_1_vae_mask_0.75_beta_1.log 2>&1 
```

# How to fine-tune?
To fine-tune the model,

```bash
  python3 run_finetune.py --dirpath lotka_d18_ed9_3_1_vae_mask_0.75_beta_1_ --num_parameters 2 > lotka_d18_ed9_3_1_finetune.log 2>&1 &
```

# How to run ABC-SMC?
```bash
  python3 run_abcsmc.py --path "lotka_d18_ed9_3_1_vae_mask_0.75_beta_1_" --tolerance_levels 0.2 0.15 0.1 0.05 0.01 --num_particles 1000 > lotka_d32_ed32_abc_mtm.log 2>&1 &
```

The implementation of LotkaVolterra class is based on 'Approximate Bayesian computation scheme for parameter inference and model selection in dynamical systems' [Toni et al., 2008].
