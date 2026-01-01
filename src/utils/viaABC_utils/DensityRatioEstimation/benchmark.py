import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial


# Import the original Python class (assuming it's in a file named original_dre.py)
# If it's in the same file, just paste the class definition here.

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


# Import the compiled C++ module
import dre_cpp

def benchmark():
    print("Generating synthetic data...")
    # Parameters for stress test
    N_SAMPLES = 2000
    N_BASIS = 100
    DIM = 10
    
    # Synthetic Data
    np.random.seed(42)
    x = np.random.randn(N_SAMPLES, DIM) + 1
    y = np.random.randn(N_SAMPLES, DIM)
    
    candidates = [0.5, 1.0, 1.5, 2.0]
    fixed_sigma = 1.0

    results = []

    print(f"\n--- BENCHMARK START (N={N_SAMPLES}, Dim={DIM}, Basis={N_BASIS}) ---\n")

    # -----------------------------
    # Test 1: Fit with Fixed Sigma
    # -----------------------------
    print("1. Testing 'fit' (Fixed Sigma)...")
    
    # Python
    py_model = DensityRatioEstimation(n=N_BASIS, max_iter=200, optimize=False)
    start = time.perf_counter()
    py_model.fit(x, y, sigma=fixed_sigma)
    py_time = time.perf_counter() - start
    results.append({'Implementation': 'Python', 'Task': 'Fit (Fixed)', 'Time': py_time})
    print(f"   Python: {py_time:.4f} s")

    # C++
    cpp_model = dre_cpp.DensityRatioEstimation(n=N_BASIS, max_iter=200, optimize=False)
    start = time.perf_counter()
    cpp_model.fit(x, y, sigma=fixed_sigma)
    cpp_time = time.perf_counter() - start
    results.append({'Implementation': 'C++', 'Task': 'Fit (Fixed)', 'Time': cpp_time})
    print(f"   C++:    {cpp_time:.4f} s")
    print(f"   >>> Speedup: {py_time / cpp_time:.1f}x")

    # -----------------------------
    # Test 2: Fit with Cross-Validation
    # -----------------------------
    print("\n2. Testing 'fit' (Cross-Validation Optimization)...")
    
    # Python
    py_model_opt = DensityRatioEstimation(n=N_BASIS, max_iter=50, fold=5, optimize=True)
    start = time.perf_counter()
    py_model_opt.fit(x, y, sigma=candidates)
    py_time_cv = time.perf_counter() - start
    results.append({'Implementation': 'Python', 'Task': 'Fit (CV)', 'Time': py_time_cv})
    print(f"   Python: {py_time_cv:.4f} s")

    # C++
    cpp_model_opt = dre_cpp.DensityRatioEstimation(n=N_BASIS, max_iter=50, fold=5, optimize=True)
    start = time.perf_counter()
    cpp_model_opt.fit(x, y, sigma=candidates)
    cpp_time_cv = time.perf_counter() - start
    results.append({'Implementation': 'C++', 'Task': 'Fit (CV)', 'Time': cpp_time_cv})
    print(f"   C++:    {cpp_time_cv:.4f} s")
    print(f"   >>> Speedup: {py_time_cv / cpp_time_cv:.1f}x")

    # -----------------------------
    # Test 3: Prediction
    # -----------------------------
    print("\n3. Testing 'predict' (Inference on 10k samples)...")
    x_test = np.random.randn(10000, DIM)
    
    # Python
    start = time.perf_counter()
    _ = py_model.predict(x_test)
    py_time_pred = time.perf_counter() - start
    results.append({'Implementation': 'Python', 'Task': 'Predict', 'Time': py_time_pred})
    print(f"   Python: {py_time_pred:.4f} s")

    # C++
    start = time.perf_counter()
    _ = cpp_model.predict(x_test)
    cpp_time_pred = time.perf_counter() - start
    results.append({'Implementation': 'C++', 'Task': 'Predict', 'Time': cpp_time_pred})
    print(f"   C++:    {cpp_time_pred:.4f} s")
    print(f"   >>> Speedup: {py_time_pred / cpp_time_pred:.1f}x")

    # -----------------------------
    # Visualization
    # -----------------------------
    df = pd.DataFrame(results)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Task', y='Time', hue='Implementation')
    plt.title('Runtime Comparison: Python (Numpy) vs C++ (Eigen)')
    plt.ylabel('Time (seconds)')
    plt.yscale('log') # Log scale because difference is likely huge
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.show()

if __name__ == "__main__":
    benchmark()