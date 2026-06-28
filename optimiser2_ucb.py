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
y_new1 = 0.143753233230569
y_new2 = 0.2967074641263785

x_new = [0.700000, 0.500000]
y_new = 0.6474771495427463

# Augment dataset
X = np.append(X, [
  [0.780000, 0.940000], 
  [0.640000, 0.950000], 
  [0.800000, 0.910000], 
  [0.700000, 0.500000], 
  [0.750000, 0.510000],
  [0.690000, 0.490000],
  [0.700000, 0.510000],
  [0.700000, 0.480000],
  [0.898834, 0.276177],
  [0.714039, 0.494616],  # week 10
  [0.715076, 0.509660]], axis=0)  # week 11
Y = np.append(Y, [
  0.143753233230569,
  0.2967074641263785,
  -0.02892579388263639,
  0.6474771495427463,
  0.36279134091949267,
  0.4391892686996867,
  0.5170222929268324,
  0.4918772545666871,
  0.06741558706200046,
  0.5944031150987877,  # week 10
  0.5216833985499837   # week 11
  ], axis=0)

print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("Y min/max:", Y.min(), Y.max())

# GP model
alpha = 1e-2
beta = 1.0   # exploration parameter for UCB


# kernel = RBF(length_scale=0.3, length_scale_bounds=(1e-2, 10.0))
kernel = Matern(length_scale=0.06, length_scale_bounds="fixed", nu=1.5)
# kernel = RationalQuadratic(length_scale=0.2, alpha=1.0)
gp = GaussianProcessRegressor(
    kernel=kernel,
    alpha=alpha,
    normalize_y=True,
    n_restarts_optimizer=10
)

gp.fit(X, Y)

print("Learned kernel:", gp.kernel_)

num_candidates = 50000

# Best still at [0.70, 0.50] — tighten search around it
lower = np.array([0.67, 0.46])
upper = np.array([0.73, 0.54])

X_candidate = np.random.uniform(lower, upper, size=(num_candidates, 2))

mu, std = gp.predict(X_candidate, return_std=True)

beta = 0.05
acquisition = mu + beta * std

# Avoid points too close to any previous observation
min_dist = 0.015
too_close = np.min(
    np.linalg.norm(X_candidate[:, None, :] - X[None, :, :], axis=2),
    axis=1
) < min_dist

acquisition[too_close] = -np.inf

best_idx = np.argmax(acquisition)
x_next = X_candidate[best_idx]

print("Best observed value:", np.max(Y))
print("Best observed point:", X[np.argmax(Y)])
print("Next query point:", x_next)
print("Distance to nearest observed:", np.min(np.linalg.norm(X - x_next, axis=1)))
print("Predicted mean there:", mu[best_idx])
print("Predicted std there:", std[best_idx])
print("Acquisition value there:", acquisition[best_idx])

# ------------------------------------------------------------
# Separate grid for plotting only
# ------------------------------------------------------------

grid_size = 101
x1_vals = np.linspace(0.0, 1.0, grid_size)
x2_vals = np.linspace(0.0, 1.0, grid_size)
X1g, X2g = np.meshgrid(x1_vals, x2_vals)

X_grid = np.column_stack([X1g.ravel(), X2g.ravel()])

mu, std = gp.predict(X_grid, return_std=True)
ucb = mu + beta * std

MU = mu.reshape(grid_size, grid_size)
STD = std.reshape(grid_size, grid_size)
UCB = ucb.reshape(grid_size, grid_size)
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
# override  : [0.7,0.5] to explore the empty space in the middle

# X shape: (14, 2)
# Y shape: (14,)
# Y min/max: -0.06562362443733738 0.6474771495427463
# Learned kernel: Matern(length_scale=0.0389, nu=1.5)
# Next query point: [0.75 0.51]
# Predicted mean there: 0.3765132960611757
# Predicted std there: 0.21939082792524203
# UCB value there: 1.4734674356873858

# X shape: (15, 2)
# Y shape: (15,)
# Y min/max: -0.06562362443733738 0.6474771495427463
# Learned kernel: Matern(length_scale=0.04, nu=1.5)
# Next query point: [0.69 0.49]
# Predicted mean there: 0.5955405827895912
# Predicted std there: 0.1110677193247918
# UCB value there: 0.706608302114383

# X shape: (16, 2)
# Y shape: (16,)
# Y min/max: -0.06562362443733738 0.6474771495427463
# Learned kernel: Matern(length_scale=0.06, nu=1.5)
# Next query point: [0.7  0.51]
# Predicted mean there: 0.6651499606574953
# Predicted std there: 0.059084708055968965
# UCB value there: 0.6769669022686892
#  
# X shape: (17, 2)
# Y shape: (17,)
# Y min/max: -0.06562362443733738 0.6474771495427463
# Learned kernel: Matern(length_scale=0.06, nu=1.5)
# Next query point: [0.71 0.5 ]
# Predicted mean there: 0.6319459091522738
# Predicted std there: 0.05149792839968087
# UCB value there: 0.6422454948322099 """

# X shape: (18, 2)
# Y shape: (18,)
# Y min/max: -0.06562362443733738 0.6474771495427463
# Learned kernel: Matern(length_scale=0.06, nu=1.5)
# Best observed value: 0.6474771495427463
# Best observed point: [0.7 0.5]
# Exploratory maximin point: [0.89883384 0.27617676]
# Distance to nearest observed point: 0.27717290616219187
# Predicted mean at exploratory point: 0.2884430590165331
# Predicted std at exploratory point: 0.2253083460946094
# UCB at exploratory point: 0.5137514051111425

# X shape: (19, 2)
# Y shape: (19,)
# Y min/max: -0.06562362443733738 0.6474771495427463
# Learned kernel: Matern(length_scale=0.06, nu=1.5)
# Best observed value: 0.6474771495427463
# Best observed point: [0.7 0.5]
# Next query point: [0.71403938 0.49461569]
# Distance to nearest observed: 0.015036449725888105
# Predicted mean there: 0.6208908812736711
# Predicted std there: 0.05888354861911281
# Acquisition value there: 0.6238350587046267
# X shape: (20, 2)
# Y shape: (20,)
# Best observed value: 0.6474771495427463
# Best observed point: [0.7  0.5]
# Next query point: [0.715076, 0.509660]
# Predicted mean there: 0.54599837
# Predicted std there: 0.051566
