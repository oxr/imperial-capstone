import numpy as np
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# Load existing observations
X0 = np.load("initial_data/function_1/initial_inputs.npy")          # shape (n, 2)
Y0 = np.load("initial_data/function_1/initial_outputs.npy").squeeze()  # shape (n,)

X = np.append(X0, [[0.82, 0.77],[0.42,0.89]], axis=0)
Y = np.append(Y0, [1.3233991793992935e-40, 9.104098992408183e-79], axis=0)

x_new = np.array([0.42,0.89])


print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("Y min/max:", Y.min(), Y.max())

# GP settings
length_scale = 0.08
alpha = 1e-3          # noise assumption: do NOT use 1e-10 here
beta = 3.0            # exploration parameter for UCB

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

# Upper Confidence Bound acquisition for maximisation
ucb = mu + beta * std

# Next query point = argmax acquisition
best_idx = np.argmax(ucb)
x_next = X_candidate[best_idx]

print("Next query point:", x_next)
print("Predicted mean there:", mu[best_idx])
print("Predicted std there:", std[best_idx])
print("UCB value there:", ucb[best_idx])

# Optional visualisation
MU = mu.reshape(grid_size, grid_size)
STD = std.reshape(grid_size, grid_size)
UCB = ucb.reshape(grid_size, grid_size)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Posterior mean
im0 = axes[0].contourf(X1g, X2g, MU, levels=30)
axes[0].scatter(X0[:, 0], X0[:, 1], c="red", s=30, label="Observed")
axes[0].scatter(x_new[0], x_new[1], c ="red", s = 80, marker = "x")
axes[0].scatter(x_next[0], x_next[1], c="white", edgecolors="black", s=100, label="Next query")
axes[0].set_title("GP posterior mean")
axes[0].set_xlabel("x1")
axes[0].set_ylabel("x2")
axes[0].legend()
plt.colorbar(im0, ax=axes[0])

# Posterior std
im1 = axes[1].contourf(X1g, X2g, STD, levels=30)
axes[1].scatter(X[:, 0], X[:, 1], c="red", s=30)
axes[1].scatter(x_new[0], x_new[1], c ="red", s = 80, marker = "x")
axes[1].scatter(x_next[0], x_next[1], c="white", edgecolors="black", s=100)
axes[1].set_title("GP posterior std")
axes[1].set_xlabel("x1")
axes[1].set_ylabel("x2")
plt.colorbar(im1, ax=axes[1])

# Acquisition
im2 = axes[2].contourf(X1g, X2g, UCB, levels=30)
axes[2].scatter(X[:, 0], X[:, 1], c="red", s=30)
axes[2].scatter(x_new[0], x_new[1], c ="red", s = 80, marker = "x")
axes[2].scatter(x_next[0], x_next[1], c="white", edgecolors="black", s=100)
axes[2].set_title(f"UCB acquisition (beta={beta})")
axes[2].set_xlabel("x1")
axes[2].set_ylabel("x2")
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.show()


# X shape: (10, 2)
# Y shape: (10,)
# Y min/max: -0.0036060626443634764 7.710875114502849e-16
# Next query point: [0.82 0.77]
# Predicted mean there: 0.0006226199273622552
# Predicted std there: 0.0009198278352188195
# UCB value there: 0.002462275597799894

# X shape: (11, 2)
# Y shape: (11,)
# Y min/max: -0.0036060626443634764 7.710875114502849e-16
# Next query point: [0.42 0.89]
# Predicted mean there: -0.00022371296079662107
# Predicted std there: 0.0010136627224347798
# UCB value there: 0.0018036124840729384

# X shape: (12, 2)
# Y shape: (12,)
# Y min/max: -0.0036060626443634764 7.710875114502849e-16
# Next query point: [0.93 0.41]
# Predicted mean there: -0.0002455524253998599
# Predicted std there: 0.0009879128496163168
# UCB value there: 0.0027181861234490904