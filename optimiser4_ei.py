import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.stats import norm

# Load data
X = np.load("initial_data/function_4/initial_inputs.npy")              # shape (n, 4)
Y = np.load("initial_data/function_4/initial_outputs.npy").squeeze()   # shape (n,)

# New observed point
x_new = np.array([0.460348, 0.439865, 0.399919, 0.434560])
y_new = 0.42194022156229805

# Add new point to dataset
X = np.append(X, [x_new], axis=0)
Y = np.append(Y, [y_new], axis=0)

print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("Y min/max:", Y.min(), Y.max())

# GP model
alpha = 1e-2
xi = 0.01

kernel = RBF(length_scale=0.2, length_scale_bounds=(1e-2, 1.0))
gp = GaussianProcessRegressor(
    kernel=kernel,
    alpha=alpha,
    normalize_y=True,
    n_restarts_optimizer=5
)

gp.fit(X, Y)

print("Learned kernel:", gp.kernel_)

# Random candidate points in 4D
num_candidates = 50000
X_candidate = np.random.uniform(0.0, 1.0, size=(num_candidates, 4))

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


#X shape: (30, 4)
#Y shape: (30,)
#Y min/max: -32.625660215962455 -4.025542281908162
#Learned kernel: RBF(length_scale=0.481)
#Best observed value: -4.025542281908162
#Next query point: [0.45810222 0.434231   0.37867107 0.44444325]
#Predicted mean there: -1.0449277189001123
#Predicted std there: 1.2189833430974326
#EI value there: 2.973579507714067

# X shape: (31, 4)
# Y shape: (31,)
# Y min/max: -32.625660215962455 0.42194022156229805
# Learned kernel: RBF(length_scale=0.491)
# Best observed value: 0.42194022156229805
# Next query point: [0.44468993 0.43866107 0.33526684 0.4391981 ]
# Predicted mean there: -0.062271532450729694
# Predicted std there: 0.7974381177171157
# EI value there: 0.13023850144030744