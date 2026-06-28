import numpy as np
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# Load existing observations
X0 = np.load("initial_data/function_1/initial_inputs.npy")              # shape (n, 2)
Y0 = np.load("initial_data/function_1/initial_outputs.npy").squeeze()   # shape (n,)

# Add your observed points
X = np.append(
    X0,
    [
        [0.82, 0.77],
        [0.42, 0.89],
        [0.50, 0.50],
        [0.350000, 0.580000],
        [0.360000, 0.380000],
        [0.760000, 0.770000],
        [0.800000, 0.690000],
        [0.900000, 0.780000],
        [0.604331, 0.284448],
        [0.439984, 0.459763],  # week 10
        [0.425839, 0.452617]   # week 11
    ],
    axis=0,
)

Y = np.append(
    Y0,
    [
        1.3233991793992935e-40,
        9.104098992408183e-79,
        2.6752879910742468e-9,
        -9.283427262667025e-22,
        0.00001028964959890712,
        1.0276453960669432e-26,
        -1.15195976794276e-23,
        -1.9049415182523047e-68,
        1.863144018630619e-37,
        0.0015757648161457936,  # week 10 — new best, massive jump
        0.06442176658113204    # week 11
    ],
    axis=0,
)

print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("Y min/max:", Y.min(), Y.max())
print("Best observed value:", np.max(Y))
print("Best observed point:", X[np.argmax(Y)])

# GP settings
length_scale = 0.08
alpha = 1e-6
beta = 0.03

kernel = RBF(length_scale=length_scale, length_scale_bounds="fixed")
gp = GaussianProcessRegressor(
    kernel=kernel,
    alpha=alpha,
    normalize_y=True,
)

# Fit GP
gp.fit(X, Y)

print("Learned kernel:", gp.kernel_)

# ------------------------------------------------------------
# Local exploitation around current best
# ------------------------------------------------------------

num_candidates = 50000

# Tight local box around the new best [0.4258, 0.4526], biased up-right
# (huge jump 1e-5 -> 0.064 came from moving up-and-right, so keep climbing)
lower = np.array([0.42, 0.45])
upper = np.array([0.52, 0.55])

X_candidate = np.random.uniform(lower, upper, size=(num_candidates, 2))

# GP posterior on local candidates
mu, std = gp.predict(X_candidate, return_std=True)

# Very mild UCB: mostly exploitation, tiny uncertainty allowance
beta = 0.03
acquisition = mu + beta * std

# Avoid exact/nearly exact repeats
tol = 1e-6
for x_obs in X:
    duplicate_mask = np.linalg.norm(X_candidate - x_obs, axis=1) < tol
    acquisition[duplicate_mask] = -np.inf

# Optional: avoid querying too close to the current best
# but keep this radius small, because F1 seems to have a narrow peak
best_x = X[np.argmax(Y)]
too_close_to_best = np.linalg.norm(X_candidate - best_x, axis=1) < 0.01
acquisition[too_close_to_best] = -np.inf

best_idx = np.argmax(acquisition)
x_next = X_candidate[best_idx]

print("Best observed value:", np.max(Y))
print("Best observed point:", best_x)
print("Next query point:", x_next)
print("Distance from best:", np.linalg.norm(x_next - best_x))
print("Predicted mean there:", mu[best_idx])
print("Predicted std there:", std[best_idx])
print("Acquisition value there:", acquisition[best_idx])

# ------------------------------------------------------------
# Separate grid for visualisation only
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

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Posterior mean
im0 = axes[0].contourf(X1g, X2g, MU, levels=30)
axes[0].scatter(X[:, 0], X[:, 1], c="red", s=30, label="Observed")
axes[0].scatter(
    x_next[0],
    x_next[1],
    c="white",
    edgecolors="black",
    s=120,
    label="Next point",
)
axes[0].scatter(
    X[np.argmax(Y), 0],
    X[np.argmax(Y), 1],
    c="yellow",
    edgecolors="black",
    s=120,
    marker="*",
    label="Best observed",
)
axes[0].set_title("GP posterior mean")
axes[0].set_xlabel("x1")
axes[0].set_ylabel("x2")
axes[0].legend()
plt.colorbar(im0, ax=axes[0])

# Posterior std
im1 = axes[1].contourf(X1g, X2g, STD, levels=30)
axes[1].scatter(X[:, 0], X[:, 1], c="red", s=30)
axes[1].scatter(
    x_next[0],
    x_next[1],
    c="white",
    edgecolors="black",
    s=120,
)
axes[1].scatter(
    X[np.argmax(Y), 0],
    X[np.argmax(Y), 1],
    c="yellow",
    edgecolors="black",
    s=120,
    marker="*",
)
axes[1].set_title("GP posterior std")
axes[1].set_xlabel("x1")
axes[1].set_ylabel("x2")
plt.colorbar(im1, ax=axes[1])

# UCB acquisition, for comparison only
im2 = axes[2].contourf(X1g, X2g, UCB, levels=30)
axes[2].scatter(X[:, 0], X[:, 1], c="red", s=30)
axes[2].scatter(
    x_next[0],
    x_next[1],
    c="white",
    edgecolors="black",
    s=120,
    label="Next point",
)
axes[2].scatter(
    X[np.argmax(Y), 0],
    X[np.argmax(Y), 1],
    c="yellow",
    edgecolors="black",
    s=120,
    marker="*",
    label="Best observed",
)
axes[2].set_title(f"UCB surface for comparison, beta={beta}")
axes[2].set_xlabel("x1")
axes[2].set_ylabel("x2")
axes[2].legend()
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

# X shape: (13, 2)
# Y shape: (13,)
# Y min/max: -0.0036060626443634764 2.6752879910742468e-09
# Next query point: [0.35 0.58]
# Predicted mean there: -0.00022653193415255927
# Predicted std there: 0.0009533740630368024
# UCB value there: 0.002633590254957848

# X shape: (14, 2)
# Y shape: (14,)
# Y min/max: -0.0036060626443634764 2.6752879910742468e-09
# Next query point: [0.36 0.38]
# Predicted mean there: -0.00022436835919218713
# Predicted std there: 0.0009257163871954366
# UCB value there: 0.004404213576784995

# X shape: (15, 2)
# Y shape: (15,)
# Y min/max: -0.0036060626443634764 1.028964959890712e-05
# Next query point: [0.76 0.77]
# Predicted mean there: 0.0005642284972880961
# Predicted std there: 0.0002799953833692126
# UCB value there: 0.0005922280356250173


# X shape: (16, 2)
# Y shape: (16,)
# Y min/max: -0.0036060626443634764 1.028964959890712e-05
# Next query point: [0.8  0.69]
# Predicted mean there: 0.0007663623589413033
# Predicted std there: 0.0005417967063322063
# UCB value there: 0.0008205420295745239

# X shape: (17, 2)
# Y shape: (17,)
# Y min/max: -0.0036060626443634764 1.028964959890712e-05
# Next query point: [0.9  0.78]
# Predicted mean there: 0.0002624615110179405
# Predicted std there: 0.0005622693525606504
# UCB value there: 0.0003186884462740055

# X shape: (18, 2)
# Y shape: (18,)
# Y min/max: -0.0036060626443634764 1.028964959890712e-05
# Best observed value: 1.028964959890712e-05
# Best observed point: [0.36 0.38]
# Learned kernel: RBF(length_scale=0.08)
# Exploratory maximin point: [0.60433092 0.28444816]
# Distance to nearest observed point: 0.23684459053456336
# Predicted mean at exploratory point: -0.00019179637445154027
# Predicted std at exploratory point: 0.000825958834776567
# UCB at exploratory point: -0.00010920049097388356

# X shape: (19, 2)
# Y shape: (19,)
# Y min/max: -0.0036060626443634764 1.028964959890712e-05
# Best observed value: 1.028964959890712e-05
# Best observed point: [0.36 0.38]
# Learned kernel: RBF(length_scale=0.08)
# Best observed value: 1.028964959890712e-05
# Best observed point: [0.36 0.38]
# Next query point: [0.43998391 0.45976275]
# Distance from best: 0.11295804956928539
# Predicted mean there: 5.757582358995473e-05
# Predicted std there: 0.0005371461106209221
# Acquisition value there: 7.36902069085824e-05
# X shape: (20, 2)
# Y shape: (20,)
# Best observed value: 0.0015757648161457936  (week 10 — huge jump)
# Best observed point: [0.439984 0.459763]
# Next query point: [0.425839, 0.452617]
# Predicted mean there: 0.00164409
# Predicted std there: 0.000091
