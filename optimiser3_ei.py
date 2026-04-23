import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.stats import norm

# Load data
X0 = np.load("initial_data/function_3/initial_inputs.npy")
Y0 = np.load("initial_data/function_3/initial_outputs.npy").squeeze()

x_new1 = np.array([0.375000, 0.416667, 0.458333])
y_new1 = 0.026529173868188035

x_new2 = np.array([0.416667, 0.250000, 0.500000])
y_new2 = -0.04248359577869199

x_new = x_new2
y_new = y_new2


X = np.append(X0, [x_new1, x_new2], axis=0)
Y = np.append(Y0, [y_new1, y_new2], axis=0)



X = np.append(X,[x_new], axis=0)
Y = np.append(Y,[y_new], axis=0)

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

# 3D candidate grid over [0,1]^3
grid_size = 25
x1_vals = np.linspace(0.0, 1.0, grid_size)
x2_vals = np.linspace(0.0, 1.0, grid_size)
x3_vals = np.linspace(0.0, 1.0, grid_size)

X1g, X2g, X3g = np.meshgrid(x1_vals, x2_vals, x3_vals, indexing="ij")
X_candidate = np.column_stack([X1g.ravel(), X2g.ravel(), X3g.ravel()])

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

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection="3d")

sc = ax.scatter(
    X0[:, 0], X0[:, 1], X0[:, 2],
    c=Y0, cmap="viridis", s=50, label="Original data"
)

ax.scatter(
    x_new[0], x_new[1], x_new[2],
    c="red", s=120, marker="x", label="New point"
)

ax.scatter(
    x_next[0], x_next[1], x_next[2],
    c="white", edgecolors="black", s=120, label="Next query"
)

ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("x3")
ax.set_title("3D input space with Y encoded by colour")
ax.legend()

cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Y value")

plt.tight_layout()
plt.show()


#X shape: (15, 3)
#Y shape: (15,)
#Y min/max: -0.3989255131463011 -0.034835313350078584
#Learned kernel: RBF(length_scale=0.192)
#Best observed value: -0.034835313350078584
#Next query point: [0.375      0.41666667 0.45833333]
#Predicted mean there: -0.027007692778697606
#Predicted std there: 0.060746079927152294
#EI value there: 0.023163484770095156

# X shape: (17, 3)
# Y shape: (17,)
# Y min/max: -0.3989255131463011 0.026529173868188035
# Learned kernel: RBF(length_scale=0.225)
# Best observed value: 0.026529173868188035
# Next query point: [0.41666667 0.25       0.5       ]
# Predicted mean there: 0.0045226375235165595
# Predicted std there: 0.053173478009149566
# EI value there: 0.00894085158797539


# X shape: (18, 3)
# Y shape: (18,)
# Y min/max: -0.3989255131463011 0.026529173868188035
# Learned kernel: RBF(length_scale=0.209)
# Best observed value: 0.026529173868188035
# Next query point: [0.41666667 0.54166667 0.54166667]
# Predicted mean there: 0.005455548498303403
# Predicted std there: 0.044500257302222734
# EI value there: 0.00637671306612132