import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.stats import norm

# Load data
X = np.load("initial_data/function_6/initial_inputs.npy")              # shape (n, 5)
Y = np.load("initial_data/function_6/initial_outputs.npy").squeeze()   # shape (n,)

# Add observed points
X = np.append(
    X,
    [
        [0.442313, 0.368016, 0.479441, 0.695979, 0.113930],
        [0.288449, 0.338695, 0.335482, 0.785115, 0.008604],
        [0.405747, 0.345410, 0.545632, 0.852504, 0.194695],
        [0.396553, 0.380843, 0.678443, 0.709158, 0.084207],
        [0.322244, 0.100853, 0.781676, 0.617864, 0.047923],
        [0.438108, 0.271559, 0.611112, 0.782106, 0.033959],
        [0.502990, 0.325583, 0.655442, 0.730069, 0.145543],
        [0.462430, 0.437177, 0.615402, 0.804091, 0.057163],
        [0.418992, 0.347828, 0.606262, 0.710744, 0.070348],   # week 11
    ],
    axis=0,
)

Y = np.append(
    Y,
    [
        -0.25949977908512906,
        -0.5641101270576772,
        -0.29623880366895267,
        -0.17467036435627814,
        -0.7149141061772797,
        -0.2031829226152843,
        -0.20629375405125805,
        -0.26991074179681634,
        -0.1273248497587954,    # week 11
    ],
    axis=0,
)

print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("Y min/max:", Y.min(), Y.max())

# GP settings
alpha = 1e-6
xi = 0.0

kernel = RBF(length_scale=0.25, length_scale_bounds="fixed")

gp = GaussianProcessRegressor(
    kernel=kernel,
    alpha=alpha,
    normalize_y=True,
)

gp.fit(X, Y)

print("Learned kernel:", gp.kernel_)

# Candidate points in tightened local 5D box
num_candidates = 80000

# recentred on new best [0.419, 0.348, 0.606, 0.711, 0.070] (week 11)
lower = np.array([0.37, 0.30, 0.56, 0.66, 0.03])
upper = np.array([0.47, 0.40, 0.66, 0.76, 0.12])

X_candidate = np.random.uniform(lower, upper, size=(num_candidates, 5))

# GP posterior
mu, std = gp.predict(X_candidate, return_std=True)

# Expected Improvement for maximisation
best_y = np.max(Y)
improvement = mu - best_y - xi

ei = np.zeros_like(mu)
mask = std > 1e-12

Z = np.zeros_like(mu)
Z[mask] = improvement[mask] / std[mask]

ei[mask] = (
    improvement[mask] * norm.cdf(Z[mask])
    + std[mask] * norm.pdf(Z[mask])
)

# Avoid points too close to any previous observation
min_dist = 0.02

dist_to_nearest_observed = np.min(
    np.linalg.norm(X_candidate[:, None, :] - X[None, :, :], axis=2),
    axis=1,
)

ei[dist_to_nearest_observed < min_dist] = -np.inf

# Select next query
best_idx = np.argmax(ei)
x_next = X_candidate[best_idx]

print("Best observed value:", best_y)
print("Best observed point:", X[np.argmax(Y)])
print("Next query point:", x_next)
print("Distance from best:", np.linalg.norm(x_next - X[np.argmax(Y)]))
print("Distance to nearest observed:", dist_to_nearest_observed[best_idx])
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

# X shape: (22, 5)
# Y shape: (22,)
# Y min/max: -2.5711696316081234 -0.25949977908512906
# Learned kernel: RBF(length_scale=0.482)
# Best observed value: -0.25949977908512906
# Next query point: [0.40574741 0.34540983 0.54563219 0.85250392 0.1946953 ]
# Predicted mean there: -0.33588726948447234
# Predicted std there: 0.16650272512461106
# EI value there: 0.031976418022587834

# X shape: (24, 5)
# Y shape: (24,)
# Y min/max: -2.5711696316081234 -0.17467036435627814
# Learned kernel: RBF(length_scale=0.528)
# Best observed value: -0.17467036435627814
# Next query point: [0.32224398 0.10085276 0.78167556 0.61786364 0.04792335]
# Predicted mean there: -0.3759559851394625
# Predicted std there: 0.2397209335864598
# EI value there: 0.026655773559717932

# X shape: (25, 5)
# Y shape: (25,)
# Y min/max: -2.5711696316081234 -0.17467036435627814
# Learned kernel: RBF(length_scale=0.25)
# Best observed value: -0.17467036435627814
# Next query point: [0.43810795 0.27155935 0.611112   0.78210629 0.03395876]
# Predicted mean there: -0.219656520638418
# Predicted std there: 0.28148173151057904
# EI value there: 0.09123296800543565

# X shape: (26, 5)
# Y shape: (26,)
# Y min/max: -2.5711696316081234 -0.17467036435627814
# Learned kernel: RBF(length_scale=0.25)
# Best observed value: -0.17467036435627814
# Next query point: [0.50298965 0.32558313 0.655442   0.73006947 0.14554296]
# Predicted mean there: -0.20936639118690858
# Predicted std there: 0.2202321835043412
# EI value there: 0.07159999856752823

# X shape: (27, 5)
# Y shape: (27,)
# Y min/max: -2.5711696316081234 -0.17467036435627814
# Learned kernel: RBF(length_scale=0.25)
# Best observed value: -0.17467036435627814
# Next query point: [0.46243026 0.43717672 0.61540199 0.80409089 0.05716348]
# Predicted mean there: -0.24513029365919203
# Predicted std there: 0.2593231880867573
# EI value there: 0.0720204650676442

# X shape: (28, 5)
# Y shape: (28,)
# Y min/max: -2.5711696316081234 -0.17467036435627814
# Learned kernel: RBF(length_scale=0.25)
# Best observed value: -0.17467036435627814
# Best observed point: [0.396553 0.380843 0.678443 0.709158 0.084207]
# Next query point: [0.37776266 0.3309646  0.62279504 0.74415646 0.11212825]
# Distance from best: 0.08911854702546898
# Distance to nearest observed: 0.08911854702546898
# Predicted mean there: -0.14227185934626196
# Predicted std there: 0.09593664420459988
# EI value there: 0.05663438450263041