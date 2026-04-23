import numpy as np
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.stats import norm

# Load data
X = np.load("initial_data/function_2/initial_inputs.npy")              # shape (n, 2)
Y = np.load("initial_data/function_2/initial_outputs.npy").squeeze()   # shape (n,)

# New observed point
x_new1 = np.array([0.780000, 0.940000])
x_new2= np.array([0.640000, 0.950000])
y_new1 = 0.143753233230569
y_new2 = 0.2967074641263785

# Augment dataset
X = np.append(X, [x_new1, x_new2], axis=0)
Y = np.append(Y, [y_new1, y_new2], axis=0)

print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("Y min/max:", Y.min(), Y.max())

x_new= x_new2
y_new = y_new2

# GP model
alpha = 1e-2
xi = 0.01   # exploration parameter for EI

kernel = RBF(length_scale=0.2, length_scale_bounds=(1e-2, 1.0))
gp = GaussianProcessRegressor(
    kernel=kernel,
    alpha=alpha,
    normalize_y=True,
    n_restarts_optimizer=5
)

gp.fit(X, Y)

print("Learned kernel:", gp.kernel_)

# Candidate grid
grid_size = 101
x1_vals = np.linspace(0.0, 1.0, grid_size)
x2_vals = np.linspace(0.0, 1.0, grid_size)
X1g, X2g = np.meshgrid(x1_vals, x2_vals)
X_candidate = np.column_stack([X1g.ravel(), X2g.ravel()])

# GP posterior
mu, std = gp.predict(X_candidate, return_std=True)

# Best observed value so far
best_y = np.max(Y)

# Expected Improvement
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

# Plot
MU = mu.reshape(grid_size, grid_size)
STD = std.reshape(grid_size, grid_size)
EI = ei.reshape(grid_size, grid_size)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

im0 = axes[0].contourf(X1g, X2g, MU, levels=30)
axes[0].scatter(X[:, 0], X[:, 1], c="red", s=35)
axes[0].scatter(x_next[0], x_next[1], c="white", edgecolors="black", s=120)
axes[0].set_title("GP posterior mean")
axes[0].set_xlabel("x1")
axes[0].set_ylabel("x2")
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].contourf(X1g, X2g, STD, levels=30)
axes[1].scatter(X[:, 0], X[:, 1], c="red", s=35)
axes[1].scatter(x_next[0], x_next[1], c="white", edgecolors="black", s=120)
axes[1].set_title("GP posterior std")
axes[1].set_xlabel("x1")
axes[1].set_ylabel("x2")
plt.colorbar(im1, ax=axes[1])

im2 = axes[2].contourf(X1g, X2g, EI, levels=30)
axes[2].scatter(X[:, 0], X[:, 1], c="red", s=35)
axes[2].scatter(x_next[0], x_next[1], c="white", edgecolors="black", s=120)
axes[2].set_title(f"Expected Improvement (xi={xi})")
axes[2].set_xlabel("x1")
axes[2].set_ylabel("x2")
plt.colorbar(im2, ax=axes[2])

# highlight newly added point
axes[0].scatter(x_new[0], x_new[1], c="red", s=90, marker="x", label="New point")
axes[1].scatter(x_new[0], x_new[1], c="red", s=90, marker="x")
axes[2].scatter(x_new[0], x_new[1], c="red", s=90, marker="x")

plt.tight_layout()
plt.show()


# X shape: (10, 2)
# Y shape: (10,)
# Y min/max: -0.06562362443733738 0.6112052157614438
# Learned kernel: RBF(length_scale=0.125)
# Best observed value: 0.6112052157614438
# Next query point: [0.78 0.94]
# Predicted mean there: 0.6224730546321133
# Predicted std there: 0.11717612922094865
# EI value there: 0.0473831679422877