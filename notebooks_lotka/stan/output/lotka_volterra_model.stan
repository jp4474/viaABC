functions {
  array[] real dz_dt(real t,       // time
                     array[] real z,     // system state {prey, predator}
                     array[] real theta, // parameters
                     array[] real x_r,   // unused data
                     array[] int x_i) {
    real u = z[1];
    real v = z[2];

    real alpha = theta[1];
    real beta = 1;
    real gamma = 1;
    real delta = theta[2];

    real du_dt = (alpha - beta * v) * u;
    real dv_dt = (-gamma + delta * u) * v;

    return { du_dt, dv_dt };
  }
}
data {
  int<lower = 0> N;          // number of measurement times
  array[N] real ts;                // measurement times > 0
  array[2] real y_init;     // initial measured populations
  array[N, 2] real<lower = 0> y;       // measured populations
}
parameters {
  array[2] real<lower = 0> theta;   // { alpha, delta }
  array[2] real<lower = 0> z_init;  // initial population
  array[2] real<lower = 0> sigma;   // measurement errors
}
transformed parameters {
  array[N, 2] real z = ode_rk45(dz_dt, z_init, 0, ts, theta, rep_array(0.0, 0), rep_array(0, 0), 1e-5, 1e-3, 5e2);
}
model {
  theta[1] ~ uniform(0, 10);
  theta[2] ~ uniform(0, 10);
  sigma ~ lognormal(-1, 1);
  z_init ~ lognormal(log(10), 1);
  for (k in 1:2) {
    y_init[k] ~ lognormal(log(z_init[k]), sigma[k]);
    y[ , k] ~ lognormal(log(z[, k]), sigma[k]);
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