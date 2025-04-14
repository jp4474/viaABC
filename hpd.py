"""
This code was taken from the PyMC library https://github.com/pymc-devs/pymc
"""

import numpy as np

def calc_min_interval(x, alpha):
    """Internal method to determine the minimum interval of a given width
    Assumes that x is sorted numpy array.
    """

    n = len(x)
    cred_mass = 1.0 - alpha

    interval_idx_inc = int(np.floor(cred_mass * n))
    n_intervals = n - interval_idx_inc
    interval_width = x[interval_idx_inc:] - x[:n_intervals]

    if len(interval_width) == 0:
        raise ValueError('Too few elements for interval calculation')

    min_idx = np.argmin(interval_width)
    hdi_min = x[min_idx]
    hdi_max = x[min_idx + interval_idx_inc]
    return hdi_min, hdi_max

def calc_min_interval_weighted(x, weights, alpha):
    """Internal method to determine the minimum interval of a given width with weights
    Assumes that x is a sorted numpy array.
    """

    n = len(x)
    cred_mass = 1.0 - alpha

    if weights is not None:
        if len(x) != len(weights):
            raise ValueError('x and weights must have the same length')
        # Normalize weights to sum to 1
        weights = np.array(weights)
        weights = weights / np.sum(weights)
    else:
        # If no weights are provided, use uniform weights
        weights = np.ones(n) / n

    # Compute the cumulative sum of weights to determine the weighted interval
    cum_weights = np.cumsum(weights)
    interval_idx_inc = np.searchsorted(cum_weights, cred_mass)

    n_intervals = n - interval_idx_inc
    interval_width = x[interval_idx_inc:] - x[:n_intervals]

    if len(interval_width) == 0:
        raise ValueError('Too few elements for interval calculation')

    min_idx = np.argmin(interval_width)
    hdi_min = x[min_idx]
    hdi_max = x[min_idx + interval_idx_inc]
    return hdi_min, hdi_max


def hpd(x, weights = None, alpha=0.05):
    """Calculate highest posterior density (HPD) of array for given alpha. 
    The HPD is the minimum width Bayesian credible interval (BCI).
    :Arguments:
        x : Numpy array
        An array containing MCMC samples
        alpha : float
        Desired probability of type I error (defaults to 0.05)
    """

    # Make a copy of trace
    x = x.copy()
    
    # For multivariate node
    if x.ndim > 1:
        # Transpose first, then sort
        tx = np.transpose(x, list(range(x.ndim))[1:] + [0])
        dims = np.shape(tx)
        # Container list for intervals
        intervals = np.zeros(dims[:-1] + (2,))

        for index in np.ndindex(dims[:-1]):
            # Sort trace
            sx = np.sort(tx[index])
            # Append to list
            intervals[index] = calc_min_interval_weighted(sx, weights, alpha)

        # Transpose back before returning
        return np.array(intervals)
    else:
        # Sort univariate node
        sx = np.sort(x)
        return np.array(calc_min_interval(sx, alpha))
