import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.stats import norm

# Load data
X = np.load("initial_data/function_6/initial_inputs.npy")              # shape (n, 5)
Y = np.load("initial_data/function_6/initial_outputs.npy").squeeze()   # shape (n,)

# New observed point (6D)
x_new = np.array([0.442313, 0.368016, 0.479441, 0.695979, 0.113930])
y_new = -0.25949977908512906

# Add to dataset
X = np.append(X, [x_new], axis=0)
Y = np.append(Y, [y_new], axis=0)


print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("Y min/max:", Y.min(), Y.max())

# GP settings
alpha = 1e-3
xi = 0.01

kernel = RBF(length_scale=0.25, length_scale_bounds=(1e-2, 2.0))
gp = GaussianProcessRegressor(
    kernel=kernel,
    alpha=alpha,
    normalize_y=True,
    n_restarts_optimizer=5
)

gp.fit(X, Y)

print("Learned kernel:", gp.kernel_)

# Random candidate points in 5D
num_candidates = 100000
X_candidate = np.random.uniform(0.0, 1.0, size=(num_candidates, 5))

# GP posterior
mu, std = gp.predict(X_candidate, return_std=True)

# Expected Improvement for maximisation
best_y = np.max(Y)
improvement = mu - best_y - xi

ei = np.zeros_like(mu)
mask = std > 1e-12

Z = np.zeros_like(mu)
Z[mask] = improvement[mask] / std[mask]

ei[mask] = improvement[mask] * norm.cdf(Z[mask]) + std[mask] * norm.pdf(Z[mask])

# Next query point
best_idx = np.argmax(ei)
x_next = X_candidate[best_idx]

print("Best observed value:", best_y)
print("Next query point:", x_next)
print("Predicted mean there:", mu[best_idx])
print("Predicted std there:", std[best_idx])
print("EI value there:", ei[best_idx])

# X shape: (20, 5)
# Y shape: (20,)
# Y min/max: -2.5711696316081234 -0.7142649478202404
# Learned kernel: RBF(length_scale=0.445)
# Best observed value: -0.7142649478202404
# Next query point: [0.46601126 0.33042245 0.44726049 0.75436727 0.12438702]
# Predicted mean there: -0.5396801314145899
# Predicted std there: 0.21983466362319695
# EI value there: 0.19348589941537347

# (base) Mac-mini:imperial-capstone oxr$ /opt/anaconda3/envs/ml-course/bin/python /Users/oxr/src/python/imperial-capstone/optimiser6_ei.py
# X shape: (21, 5)
# Y shape: (21,)
# Y min/max: -2.5711696316081234 -0.25949977908512906
# Learned kernel: RBF(length_scale=0.51)
# Best observed value: -0.25949977908512906
# Next query point: [0.2884485  0.33869525 0.33548247 0.78511548 0.00860379]
# Predicted mean there: -0.3099345995582583
# Predicted std there: 0.18683501204220893
# EI value there: 0.04818471488581785