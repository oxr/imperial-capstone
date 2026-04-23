import numpy as np
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import RationalQuadratic



# Load data
X = np.load("initial_data/function_2/initial_inputs.npy")              # shape (n, 2)
Y = np.load("initial_data/function_2/initial_outputs.npy").squeeze()   # shape (n,)

# New observed point
x_new1 = np.array([0.780000, 0.940000])
x_new2 = np.array([0.640000, 0.950000])
y_new1 = 0.143753233230569
y_new2 = 0.2967074641263785

x_new = x_new2
y_new = y_new2

# Augment dataset
X = np.append(X, [x_new1, x_new2], axis=0)
Y = np.append(Y, [y_new1, y_new2], axis=0)

print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("Y min/max:", Y.min(), Y.max())

# GP model
alpha = 1e-2
beta = 5.0   # exploration parameter for UCB


# kernel = RBF(length_scale=0.3, length_scale_bounds=(1e-2, 10.0))
kernel = Matern(length_scale=0.125, length_scale_bounds=(0.05, 1.0), nu=1.5)
# kernel = RationalQuadratic(length_scale=0.2, alpha=1.0)
gp = GaussianProcessRegressor(
    kernel=kernel,
    alpha=alpha,
    normalize_y=True,
    n_restarts_optimizer=10
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

# Upper Confidence Bound
ucb = mu + beta * std

# Next query point
best_idx = np.argmax(ucb)
x_next = X_candidate[best_idx]

print("Next query point:", x_next)
print("Predicted mean there:", mu[best_idx])
print("Predicted std there:", std[best_idx])
print("UCB value there:", ucb[best_idx])

# Plot
MU = mu.reshape(grid_size, grid_size)
STD = std.reshape(grid_size, grid_size)
UCB = ucb.reshape(grid_size, grid_size)

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


im2 = axes[2].contourf(X1g, X2g, UCB, levels=30)
axes[2].scatter(X[:, 0], X[:, 1], c="red", s=35)
axes[2].scatter(x_new[0], x_new[1], c="red", s=90, marker="x")
axes[2].scatter(x_next[0], x_next[1], c="white", edgecolors="black", s=120)
axes[2].set_title(f"UCB acquisition (beta={beta})")
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


# Learned kernel: Matern(length_scale=0.05, nu=1.5)
# Next query point: [0.64 0.95]
# Predicted mean there: 0.3530186693933074
# Predicted std there: 0.20440457780215188
# UCB value there: 1.375041558404067


# Since only a small number of observations were available, 
# unconstrained hyperparameter fitting drove the RBF length 
# scale to a very small value, producing an overly local 
# surrogate and discouraging broader exploration. 
# I therefore imposed a lower bound on the length scale 
# to maintain a smoother model, reflecting the view that 
# at this early stage the optimiser should still reason at 
# a coarser spatial scale rather than fit fine 
# local variation.


# week3 query
# X shape: (11, 2)
# Y shape: (11,)
# Y min/max: -0.06562362443733738 0.6112052157614438
# Learned kernel: Matern(length_scale=0.0734, nu=1.5)
# Next query point: [0.8  0.91]
# Predicted mean there: 0.38980047666988354
# Predicted std there: 0.20190918800313687
# UCB value there: 1.3993464166855678
