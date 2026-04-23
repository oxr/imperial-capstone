import numpy as np
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.stats import norm

# Load existing observations
X0 = np.load("initial_data/function_1/initial_inputs.npy")             # shape (n, 2)
Y0 = np.load("initial_data/function_1/initial_outputs.npy").squeeze()  # shape (n,)

# Add the new observed point
x_new = np.array([0.82, 0.77])
x_new2 = np.array([0.42, 0.89])
y_new = 1.3233991793992935e-40
y_new2 = 9.104098992408183e-79
X = np.append(X0, [x_new, x_new2], axis=0)
Y = np.append(Y0, [y_new, y_new2], axis=0)

print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("Y min/max:", Y.min(), Y.max())

# GP settings
length_scale = 0.08
alpha = 1e-3
xi = 0.01   # exploration parameter for EI

kernel = RBF(length_scale=length_scale, length_scale_bounds="fixed")
gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=True)

# Fit GP
gp.fit(X, Y)

# Candidate grid over the 2D search area
grid_size = 101
x1_vals = np.linspace(0.0, 1.0, grid_size)
x2_vals = np.linspace(0.0, 1.0, grid_size)
X1g, X2g = np.meshgrid(x1_vals, x2_vals)

X_candidate = np.column_stack([X1g.ravel(), X2g.ravel()])

# GP posterior on candidate grid
mu, std = gp.predict(X_candidate, return_std=True)

# Expected Improvement for maximisation
f_best = np.max(Y)

ei = np.zeros_like(mu)
mask = std > 1e-12

z = (mu[mask] - f_best - xi) / std[mask]
ei[mask] = (mu[mask] - f_best - xi) * norm.cdf(z) + std[mask] * norm.pdf(z)

# Next query point = argmax EI
best_idx = np.argmax(ei)
x_next = X_candidate[best_idx]

print("Best observed value:", f_best)
print("Next query point:", x_next)
print("Predicted mean there:", mu[best_idx])
print("Predicted std there:", std[best_idx])
print("EI value there:", ei[best_idx])

# Reshape for plotting
MU = mu.reshape(grid_size, grid_size)
STD = std.reshape(grid_size, grid_size)
EI = ei.reshape(grid_size, grid_size)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Posterior mean
im0 = axes[0].contourf(X1g, X2g, MU, levels=30)
axes[0].scatter(X0[:, 0], X0[:, 1], c="red", s=30, label="Original data")
axes[0].scatter(x_new[0], x_new[1], c="red", s=80, marker="x", label="New point")
axes[0].scatter(x_next[0], x_next[1], c="white", edgecolors="black", s=100, label="Next query")
axes[0].set_title("GP posterior mean")
axes[0].set_xlabel("x1")
axes[0].set_ylabel("x2")
axes[0].legend()
plt.colorbar(im0, ax=axes[0])

# Posterior std
im1 = axes[1].contourf(X1g, X2g, STD, levels=30)
axes[1].scatter(X0[:, 0], X0[:, 1], c="red", s=30)
axes[1].scatter(x_new[0], x_new[1], c="red", s=80, marker="x")
axes[1].scatter(x_next[0], x_next[1], c="white", edgecolors="black", s=100)
axes[1].set_title("GP posterior std")
axes[1].set_xlabel("x1")
axes[1].set_ylabel("x2")
plt.colorbar(im1, ax=axes[1])

# EI acquisition
im2 = axes[2].contourf(X1g, X2g, EI, levels=30)
axes[2].scatter(X0[:, 0], X0[:, 1], c="red", s=30)
axes[2].scatter(x_new[0], x_new[1], c="red", s=80, marker="x")
axes[2].scatter(x_next[0], x_next[1], c="white", edgecolors="black", s=100)
axes[2].set_title(f"Expected Improvement (xi={xi})")
axes[2].set_xlabel("x1")
axes[2].set_ylabel("x2")
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.show()


# X shape: (11, 2)
# Y shape: (11,)
# Y min/max: -0.0036060626443634764 7.710875114502849e-16
# Best observed value: 7.710875114502849e-16
# Next query point: [0.25 0.28]
# Predicted mean there: -0.0003006538968472635
# Predicted std there: 0.0010352435745208507
# EI value there: 1.286907298352591e-27

# X shape: (12, 2)
# Y shape: (12,)
# Y min/max: -0.0036060626443634764 7.710875114502849e-16
# Best observed value: 7.710875114502849e-16
# Next query point: [0.25 0.28]
# Predicted mean there: -0.00027559940520172896
# Predicted std there: 0.0009952915692213653
# EI value there: 2.5925418271448894e-29