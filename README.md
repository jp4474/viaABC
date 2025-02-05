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


```bash
  python3 run_pretrain.py
```


# How to run?
```python
  Lotka = LotkaVolterra()  
  Lotka.update_model('your model') # pass in the trained model (lightning module class) from above
  particles, weights = Lotka.run()
  Lotka.compute_statistics() # examine particles and weights in the final generation

  final_posterior = particles[-1]
  final_weights = weights[-1]
```

The implementation of LotkaVolterra class is based on 'Approximate Bayesian computation scheme for parameter inference and model selection in dynamical systems' [Toni et al., 2008].
