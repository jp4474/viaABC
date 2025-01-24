# latent-abc-smc

LatentABCSMC implements a Sequential Monte Carlo (SMC) algorithm for Approximate Bayesian Computation (ABC). This Python class is designed to estimate posterior distributions of model parameters when direct likelihood calculations are infeasible due to computational constraints or the absence of explicit likelihood functions for the data. The algorithm extends the standard ABC-SMC framework by leveraging a pre-trained Variational Autoencoder (VAE) to encode data into a latent representation. This approach enables the computation of distances in the latent space, offering a more effective alternative to traditional summary statistics or conventional distance metrics.


```python
  Lotka = LotkaVolterra()
  Lotka.generate_training_data() # use this data to train a VAE
  
  Lotka.update_model('your model')
  
  particles, weights = Lotka.run()
  
  final_posterior = particles[-1]
  final_weights = weights[-1]
```

The implementation of LotkaVolterra class is based on 'Approximate Bayesian computation scheme for parameter inference and model selection in dynamical systems' [Toni et al., 2008].
