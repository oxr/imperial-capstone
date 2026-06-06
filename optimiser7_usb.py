import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# Load data
X = np.load("initial_data/function_7/initial_inputs.npy")              # shape (n, 6)
Y = np.load("initial_data/function_7/initial_outputs.npy").squeeze()   # shape (n,)

# Add observed points
X = np.append(
    X,
    [
        [0.001580, 0.391010, 0.153200, 0.245729, 0.376003, 0.791463],
        [0.033758, 0.343269, 0.718226, 0.127718, 0.389525, 0.914858],
        [0.034991, 0.092681, 0.206994, 0.090283, 0.361562, 0.729274],
        [0.005701, 0.010807, 0.151242, 0.025448, 0.268732, 0.989291],
        [0.111196, 0.070977, 0.109673, 0.041995, 0.453427, 0.816965],
        [0.000000, 0.201898, 0.595100, 0.131656, 0.307225, 0.709082],
        [0.000000, 0.273738, 0.994778, 0.039971, 0.278406, 0.688063],
        [0.000000, 0.022220, 0.413597, 0.393732, 0.226817, 0.703969],
    ],
    axis=0,
)

Y = np.append(
    Y,
    [
        1.547355474143622,
        1.34597161695071,
        1.6380180139105758,
        0.48272596897408004,
        0.8081594477521297,
        2.253579268477676,
        0.7723496274114856,
        2.0768595934217973,
    ],
    axis=0,
)

print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("Y min/max:", Y.min(), Y.max())

# ------------------------------------------------------------
# GP settings
# ------------------------------------------------------------

alpha = 1e-5
beta = 0.05   # mild UCB: mostly exploitative

kernel = RBF(
    length_scale=np.array([0.5, 0.5, 2.0, 0.5, 0.2, 0.2]),
    length_scale_bounds="fixed",
)

gp = GaussianProcessRegressor(
    kernel=kernel,
    alpha=alpha,
    normalize_y=True,
)

gp.fit(X, Y)

best_x = X[np.argmax(Y)]
best_y = np.max(Y)

print("Kernel:", gp.kernel_)
print("Current best observed value:", best_y)
print("Current best observed point:", best_x)

# ------------------------------------------------------------
# Candidate points in constrained F7 local region
# ------------------------------------------------------------
# Avoid x3 ≈ 1, since that recent query performed badly.
# Keep x1 near 0 and x6 near 0.70.
# ------------------------------------------------------------

num_candidates = 100000

lower = np.array([0.00, 0.10, 0.40, 0.08, 0.24, 0.65])
upper = np.array([0.06, 0.30, 0.70, 0.36, 0.36, 0.76])

X_candidate = np.random.uniform(lower, upper, size=(num_candidates, 6))

# GP posterior
mu, std = gp.predict(X_candidate, return_std=True)

# UCB acquisition
ucb = mu + beta * std

# ------------------------------------------------------------
# Avoid points too close to any previous observation
# ------------------------------------------------------------

min_dist = 0.03

dist_to_nearest_observed = np.min(
    np.linalg.norm(X_candidate[:, None, :] - X[None, :, :], axis=2),
    axis=1,
)

ucb[dist_to_nearest_observed < min_dist] = -np.inf

# ------------------------------------------------------------
# Select next query
# ------------------------------------------------------------

best_idx = np.argmax(ucb)
x_next = X_candidate[best_idx]

print("Next query point:", x_next)
print("Distance from best:", np.linalg.norm(x_next - best_x))
print("Distance to nearest observed:", dist_to_nearest_observed[best_idx])
print("Predicted mean there:", mu[best_idx])
print("Predicted std there:", std[best_idx])
print("UCB value there:", ucb[best_idx])

# X shape: (30, 6)
# Y shape: (30,)
# Y min/max: 0.0027014650245082332 1.3649683044991994
#/opt/anaconda3/envs/ml-course/lib/python3.12/site-packages/sklearn/gaussian_process/kernels.py:450: ConvergenceWarning: The optimal value found for dimension 2 of parameter length_scale is close to the specified upper bound 10.0. Increasing the bound and calling fit again may find a better value.
#  warnings.warn(
#Learned kernel: RBF(length_scale=[0.915, 1.74, 10, 0.36, 0.162, 0.155])
#Current best observed value: 1.3649683044991994
#Current best observed point: [0.05789554 0.49167222 0.24742222 0.21811844 0.42042833 0.73096984]
#Next query point: [0.01606775 0.01504925 0.48552772 0.25080705 0.38422449 0.7630729 ]
#Predicted mean there: 1.3238081050918642
#Predicted std there: 0.1152227518912373
#UCB value there: 1.4966422329287201

# (base) Mac-mini:imperial-capstone oxr$ /opt/anaconda3/envs/ml-course/bin/python /Users/oxr/src/python/imperial-capstone/optimiser7_usb.py
# X shape: (31, 6)
# Y shape: (31,)
# Y min/max: 0.0027014650245082332 1.547355474143622
# /opt/anaconda3/envs/ml-course/lib/python3.12/site-packages/sklearn/gaussian_process/kernels.py:450: ConvergenceWarning: The optimal value found for dimension 2 of parameter length_scale is close to the specified upper bound 10.0. Increasing the bound and calling fit again may find a better value.
#   warnings.warn(
# Learned kernel: RBF(length_scale=[0.958, 0.415, 10, 0.491, 0.271, 0.447])
# Current best observed value: 1.547355474143622
# Current best observed point: [0.00158  0.39101  0.1532   0.245729 0.376003 0.791463]
# Next query point: [0.03375804 0.34326941 0.71822644 0.12771799 0.38952518 0.91485796]
# Predicted mean there: 1.564585396466434
# Predicted std there: 0.11943942492019785
# UCB value there: 1.7437445338467308

# X shape: (32, 6)
# Y shape: (32,)
# Y min/max: 0.0027014650245082332 1.547355474143622
# Learned kernel: RBF(length_scale=[1.04, 0.628, 1.88, 0.514, 0.228, 0.462])
# Current best observed value: 1.547355474143622
# Current best observed point: [0.00158  0.39101  0.1532   0.245729 0.376003 0.791463]
# Next query point: [0.03499111 0.09268093 0.20699378 0.09028278 0.3615622  0.72927381]
# Predicted mean there: 1.4269706729056055
# Predicted std there: 0.185430209690389
# UCB value there: 1.705115987441189

# X shape: (33, 6)
# Y shape: (33,)
# Y min/max: 0.0027014650245082332 1.6380180139105758
# Learned kernel: RBF(length_scale=[1.19, 0.827, 1.48, 0.631, 0.241, 0.556])
# Current best observed value: 1.6380180139105758
# Current best observed point: [0.034991 0.092681 0.206994 0.090283 0.361562 0.729274]
# Next query point: [0.0057011  0.01080712 0.15124205 0.02544839 0.26873209 0.98929121]
# Predicted mean there: 1.5377159347784768
# Predicted std there: 0.1831299212561458
# UCB value there: 1.8124108166626955

# # X shape: (34, 6)
# Y shape: (34,)
# Y min/max: 0.0027014650245082332 1.6380180139105758
# he optimal value found for dimension 2 of parameter length_scale is close to the specified upper bound 100.0. Increasing the bound and calling fit again may find a better value.
#   warnings.warn(
# Learned kernel: RBF(length_scale=[1.06, 1.03, 100, 0.956, 0.223, 0.204])
# Current best observed value: 1.6380180139105758
# Current best observed point: [0.034991 0.092681 0.206994 0.090283 0.361562 0.729274]
# Next query point: [0.11119587 0.0709765  0.10967313 0.04199521 0.45342719 0.81696459]
# Predicted mean there: 1.629547798758057
# Predicted std there: 0.13928630426907931
# UCB value there: 1.8384772551616761

# X shape: (35, 6)
# Y shape: (35,)
# Y min/max: 0.0027014650245082332 1.6380180139105758
# Kernel: RBF(length_scale=[0.5, 0.5, 2, 0.5, 0.2, 0.2])
# Current best observed value: 1.6380180139105758
# Current best observed point: [0.034991 0.092681 0.206994 0.090283 0.361562 0.729274]
# Next query point: [0.         0.20189846 0.59510017 0.13165558 0.30722497 0.70908238]
# Distance from best: 0.4109149909928843
# Predicted mean there: 1.8545156547233812
# Predicted std there: 0.12546352361459434
# UCB value there: 1.8921547118077595

# X shape: (36, 6)
# Y shape: (36,)
# Y min/max: 0.0027014650245082332 2.253579268477676
# Kernel: RBF(length_scale=[0.5, 0.5, 2, 0.5, 0.2, 0.2])
# Current best observed value: 2.253579268477676
# Current best observed point: [0.       0.201898 0.5951   0.131656 0.307225 0.709082]
# Next query point: [0.         0.27373788 0.99477836 0.03997075 0.27840633 0.6880634 ]
# Distance from best: 0.4178303953024568
# Predicted mean there: 2.437795546690521
# Predicted std there: 0.1228294269750775
# UCB value there: 2.474644374783044

# X shape: (37, 6)
# Y shape: (37,)
# Y min/max: 0.0027014650245082332 2.253579268477676
# Kernel: RBF(length_scale=[0.5, 0.5, 2, 0.5, 0.2, 0.2])
# Current best observed value: 2.253579268477676
# Current best observed point: [0.       0.201898 0.5951   0.131656 0.307225 0.709082]
# Next query point: [0.         0.02222032 0.41359719 0.3937317  0.22681674 0.7039691 ]
# Distance from best: 0.37470341057546863
# Predicted mean there: 3.9601449705206
# Predicted std there: 0.23797480913573626
# UCB value there: 4.03153741326132

# X shape: (38, 6)
# Y shape: (38,)
# Y min/max: 0.0027014650245082332 2.253579268477676
# Kernel: RBF(length_scale=[0.5, 0.5, 2, 0.5, 0.2, 0.2])
# Current best observed value: 2.253579268477676
# Current best observed point: [0.       0.201898 0.5951   0.131656 0.307225 0.709082]
# Next query point: [5.88424470e-04 2.02819564e-01 6.89792389e-01 3.51276489e-01
#  3.51133010e-01 6.59716342e-01]
# Distance from best: 0.24812473610032804
# Distance to nearest observed: 0.24812473610032804
# Predicted mean there: 3.245882257210591
# Predicted std there: 0.2164018096469912
# UCB value there: 3.2567023476929404