import numpy as np
import pytest
from scipy.stats import multivariate_normal
from scipy.special import logsumexp

import particle_weight_cpp


def python_reference_weight(
    theta,
    prev_particles,
    prev_weights,
    prev_cov,
    prior_log_prob,
):
    """
    Exact Python reference implementation
    (matches your original code)
    """
    prior_lp = prior_log_prob(theta)

    phi = np.array([
        multivariate_normal.logpdf(
            theta,
            mean=x,
            cov=prev_cov,
            allow_singular=False,
        )
        for x in prev_particles
    ])

    log_w = np.log(prev_weights) + phi
    lse = logsumexp(log_w)
    return np.exp(prior_lp - lse)


def test_particle_weight_matches_scipy():
    rng = np.random.default_rng(0)

    # dimensions
    N = 50
    d = 4

    # generate particles
    prev_particles = rng.normal(size=(N, d))

    # strictly positive weights
    prev_weights = rng.random(N)
    prev_weights /= prev_weights.sum()
    prev_weights = np.clip(prev_weights, 1e-300, None)

    # SPD covariance
    A = rng.normal(size=(d, d))
    prev_cov = A @ A.T + 1e-6 * np.eye(d)

    # parameter to evaluate
    theta = rng.normal(size=d)

    # prior
    def prior_log_prob(theta):
        return multivariate_normal.logpdf(
            theta,
            mean=np.zeros(d),
            cov=np.eye(d),
        )

    # python reference
    w_py = python_reference_weight(
        theta,
        prev_particles,
        prev_weights,
        prev_cov,
        prior_log_prob,
    )

    # C++ implementation
    w_cpp = particle_weight_cpp.calculate_particle_weight(
        theta,
        prev_particles,
        prev_weights,
        prev_cov,
        prior_log_prob,
    )
    
    # checks
    assert np.isfinite(w_cpp)
    assert np.isfinite(w_py)

    # relative + absolute tolerance
    np.testing.assert_allclose(
        w_cpp,
        w_py,
        rtol=1e-6,   # relative tolerance
        atol=1e-12,  # absolute tolerance
    )
