functions {
  vector dz_dt(real t, vector z, array[] real theta) {
    real u = z[1];
    real v = z[2];

    real alpha = theta[1];
    real beta = 1;
    real gamma = 1;
    real delta = theta[2];

    real du_dt = (alpha - beta * v) * u;
    real dv_dt = (-gamma + delta * u) * v;

    return [du_dt, dv_dt]';
  }
}
data {
  int<lower = 0> N;                    // number of measurement times
  array[N] real ts;                    // measurement times > 0
  array[2] real y_init;                // initial measured populations
  array[N, 2] real y;                  // measured populations
}
parameters {
  array[2] real<lower = 0> theta;   // { alpha, delta }
  vector<lower = 0>[2] z_init;      // initial population
  array[2] real<lower = 0> sigma;   // measurement errors
}
transformed parameters {
  array[N] vector[2] z_solution = ode_rk45(dz_dt, z_init, 0, ts, theta);
  array[N, 2] real z;
  for (n in 1:N) {
    z[n, 1] = z_solution[n][1];
    z[n, 2] = z_solution[n][2];
  }
}
model {
  theta[1] ~ uniform(0, 10);
  theta[2] ~ uniform(0, 10);
  // z_init ~ normal(0, 0.5);
  sigma ~ normal(0, 0.5);
  for (k in 1:2) {
    y_init[k] ~ normal(z_init[k], sigma[k]);
    y[ , k] ~ normal(z[ , k], sigma[k]);
  }
}

generated quantities {
  array[2] real y_init_rep;
  array[N, 2] real y_rep;
  for (k in 1:2) {
    y_init_rep[k] = lognormal_rng(log(z_init[k]), sigma[k]);
    for (n in 1:N)
      y_rep[n, k] = lognormal_rng(log(z[n, k]), sigma[k]);
  }
}