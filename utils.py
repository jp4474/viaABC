import torch
import numpy as np
import torch.nn.functional as F
from scipy.integrate import solve_ivp

def simulator(x):
    def ode_system(t, state, parameters):
        # Lotka-Volterra equations
        alpha, delta = parameters
        beta, gamma = 1, 1
        prey, predator = state
        dprey = prey * (alpha - beta * predator)
        dpredator = predator * (-gamma + delta * prey)
        return [dprey, dpredator]

    parameters = x
    solution = solve_ivp(ode_system, t_span=[0, 15], y0=np.array([1, 0.5]), t_eval=np.array([1.1, 2.4, 3.9, 5.6, 7.5, 9.6, 11.9, 14.4]), args=(parameters,))
    return solution.y.T

# Function to generate a triplet
    #x_anchor = np.random.uniform(prior_low, prior_high, size=2)  # 2D parameter set
def generate_triplet(x_anchor, prior_low, prior_high, perturbation_scale=0.1, margin=1):
    """
    Generates an anchor, positive, and negative triplet.
    
    Args:
        prior_low (float): Lower bound for sampling parameters.
        prior_high (float): Upper bound for sampling parameters.
        perturbation_scale (float): Scale for perturbing parameters to generate positive pairs.
        margin (float): Minimum distance between anchor and negative pairs.
    
    Returns:
        anchor (tuple): (x_anchor, y_anchor)
        positive (tuple): (x_pos, y_pos)
        negative (tuple): (x_neg, y_neg)
    """
    # Generate anchor
    #x_anchor = np.random.uniform(prior_low, prior_high, size=2)  # 2D parameter set
    #y_anchor = simulator(x_anchor)
    
    # Generate positive by perturbing anchor
    while True:
        perturbation = np.random.uniform(-perturbation_scale, perturbation_scale, size=2)
        x_candidate = x_anchor + perturbation
        if np.all(x_candidate >= prior_low) and np.all(x_candidate <= prior_high):
            break
    x_pos = x_anchor + perturbation
    y_pos = simulator(x_pos)
    
    # Generate negative by sampling until it's sufficiently far from anchor
    while True:
        x_neg = np.random.uniform(prior_low, prior_high, size=2)
        if np.linalg.norm(x_neg - x_anchor) > np.linalg.norm(x_pos - x_anchor) + margin:
            break
    y_neg = simulator(x_neg)
    
    return (x_pos, y_pos), (x_neg, y_neg)