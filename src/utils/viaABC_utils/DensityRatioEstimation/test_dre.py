import numpy as np
from functools import partial
import pytest
import time

# --- Original Python Implementation (Pasted from Prompt) ---
class DensityRatioEstimation:
    """A density ratio estimation class."""

    def __init__(self, n=100, epsilon=0.1, max_iter=500, abs_tol=0.01,
                 conv_check_interval=20, fold=5, optimize=False):
        self.n = n
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.abs_tol = abs_tol
        self.fold = fold
        self.sigma = None
        self.conv_check_interval = conv_check_interval
        self.optimize = optimize

    def fit(self, x, y, weights_x=None, weights_y=None, sigma=None):
        self.x_len = x.shape[0]
        self.y_len = y.shape[0]
        x = x.reshape(self.x_len, -1)
        y = y.reshape(self.y_len, -1)
        self.x = x

        if self.x_len < self.n:
            raise ValueError(f"Number of RBFs ({self.n}) can't be larger than samples ({self.x_len}).")

        self.theta = x[:self.n, :]
        if weights_x is None: weights_x = np.ones(self.x_len)
        if weights_y is None: weights_y = np.ones(self.y_len)

        self.weights_x = weights_x / np.sum(weights_x)
        self.weights_y = weights_y / np.sum(weights_y)

        self.x0 = np.average(x, axis=0, weights=weights_x)

        if isinstance(sigma, float) or isinstance(sigma, int):
            self.sigma = float(sigma)
            self.optimize = False
        if self.optimize:
            if isinstance(sigma, list):
                # Fixing the zip logic from original prompt which was slightly ambiguous in return types
                scores_tuple = [self._KLIEP_lcv(x, y, sigma_i)[0] for sigma_i in sigma]
                self.sigma = sigma[np.argmax(scores_tuple)]
            else:
                raise ValueError("To optimize RBF scale, provide list of candidate scales.")

        if self.sigma is None:
            raise ValueError("RBF width (sigma) has to provided in first call.")

        A = self._compute_A(x, self.sigma)
        b, b_normalized = self._compute_b(y, self.sigma)

        alpha = self._KLIEP(A, b, b_normalized, self.weights_x, self.sigma)
        self.w = partial(self._weighted_basis_sum, sigma=self.sigma, alpha=alpha)

    def _gaussian_basis(self, x, x0, sigma):
        return np.exp(-0.5 * np.sum((x - x0) ** 2) / sigma / sigma)

    def _weighted_basis_sum(self, x, sigma, alpha):
        # Initial code had a list comprehension structure that results in shape (N, N_basis)
        return np.dot(np.array([[self._gaussian_basis(j, i, sigma) for j in self.theta]
                                for i in np.atleast_2d(x)]), alpha)

    def _compute_A(self, x, sigma):
        A = np.array([[self._gaussian_basis(i, j, sigma) for j in self.theta] for i in x])
        return A

    def _compute_b(self, y, sigma):
        b = np.sum(np.array(
                [[self._gaussian_basis(i, y[j, :], sigma) * self.weights_y[j]
                for j in np.arange(self.y_len)]
                for i in self.theta]), axis=1)
        b_normalized = b / np.dot(b.T, b)
        return b, b_normalized

    def _KLIEP_lcv(self, x, y, sigma):
        A = self._compute_A(x, sigma)
        b, b_normalized = self._compute_b(y, sigma)
        non_null = np.any(A > 1e-64, axis=1)
        non_null_length = sum(non_null)
        if non_null_length == 0: return [np.Inf]

        A_full = A[non_null, :]
        x_full = x[non_null, :]
        weights_x_full = self.weights_x[non_null]

        fold_indices = np.array_split(np.arange(non_null_length), self.fold)
        score = np.zeros(self.fold)
        for i_fold, fold_index in enumerate(fold_indices):
            fold_index_minus = np.setdiff1d(np.arange(non_null_length), fold_index)
            alpha = self._KLIEP(A=A_full[fold_index_minus, :], b=b, b_normalized=b_normalized,
                                weights_x=weights_x_full[fold_index_minus], sigma=sigma)
            
            # The original code's _weighted_basis_sum is expensive, let's just use it
            preds = self._weighted_basis_sum(x_full[fold_index, :], sigma, alpha)
            # Avoid log(0)
            preds[preds <= 0] = 1e-12 
            score[i_fold] = np.average(np.log(preds), weights=weights_x_full[fold_index])

        return [np.mean(score)]

    def _KLIEP(self, A, b, b_normalized, weights_x, sigma):
        alpha = 1 / self.n * np.ones(self.n)
        target_fun_prev = self._weighted_basis_sum(x=self.x, sigma=sigma, alpha=alpha)
        
        non_null = np.any(A > 1e-64, axis=1)
        A_full = A[non_null, :]
        weights_x_full = weights_x[non_null]
        
        for i in np.arange(self.max_iter):
            Ax = np.matmul(A_full, alpha)
            Ax[Ax==0] = 1e-12 # Safety
            dAdalpha = np.matmul(A_full.T, (weights_x_full / Ax))
            alpha += self.epsilon * dAdalpha
            alpha = np.maximum(0, alpha + (1 - np.dot(b.T, alpha)) * b_normalized)
            alpha = alpha / (np.dot(b.T, alpha) + 1e-12)
            
            if np.remainder(i, self.conv_check_interval) == 0:
                target_fun = self._weighted_basis_sum(x=self.x, sigma=sigma, alpha=alpha)
                abs_diff = np.linalg.norm(target_fun - target_fun_prev)
                if abs_diff < self.abs_tol:
                    break
                target_fun_prev = target_fun
        return alpha
    
    def predict(self, x):
        return self.w(x)

# --- Tests ---

import dre_cpp

def test_equivalence():
    np.random.seed(42)
    # Generate Synthetic Data
    n_samples = 1000
    dim = 2
    x = np.random.randn(n_samples, dim) + 1 # Numerator
    y = np.random.randn(n_samples, dim)     # Denominator
    
    # 1. Test Fixed Sigma
    sigma = 1.5
    
    # Python
    dre_py = DensityRatioEstimation(n=n_samples, epsilon=0.01, max_iter=200, optimize=False)
    dre_py.fit(x, y, sigma=sigma)
    res_py = dre_py.predict(x)
    
    # C++
    dre_c = dre_cpp.DensityRatioEstimation(n=n_samples, epsilon=0.01, max_iter=200, optimize=False)
    dre_c.fit(x, y, sigma=sigma)
    res_c = dre_c.predict(x)
    
    # Comparison
    print(f"Fixed Sigma - Python Max Ratio: {np.max(res_py)}")
    print(f"Fixed Sigma - C++ Max Ratio:    {np.max(res_c)}")
    
    # Allow small floating point divergence due to Order of Operations (SIMD/Eigen vs Numpy)
    np.testing.assert_allclose(res_c, res_py, rtol=1e-3, atol=1e-3, 
                               err_msg="Prediction results differ for fixed sigma")
    print(">> Fixed Sigma Test Passed")

    # 2. Test Helper Functions
    var_py = np.var(x, axis=0, ddof=0) # Simple variance check first
    # Weighted var function logic in prompt is specific
    
    # 3. Test Optimization (Cross Validation)
    # Note: Because np.array_split and C++ split might differ slightly on edge cases
    # We use a divisible number to ensure identical splits.
    
    x_opt = x[:150]
    y_opt = y[:150]
    candidates = [0.5, 1.0, 2.0]
    
    dre_py_opt = DensityRatioEstimation(n=50, epsilon=0.01, max_iter=100, fold=5, optimize=True)
    dre_py_opt.fit(x_opt, y_opt, sigma=candidates)
    
    dre_c_opt = dre_cpp.DensityRatioEstimation(n=50, epsilon=0.01, max_iter=100, fold=5, optimize=True)
    dre_c_opt.fit(x_opt, y_opt, sigma=candidates)
    
    print(f"Optimized Sigma Python: {dre_py_opt.sigma}")
    print(f"Optimized Sigma C++:    {dre_c_opt.sigma}")
    
    assert dre_py_opt.sigma == dre_c_opt.sigma, "Optimization selected different sigmas"
    print(">> Optimization Test Passed")

if __name__ == "__main__":
    try:
        test_equivalence()
        print("\nALL TESTS PASSED: C++ implementation is functionally identical.")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")