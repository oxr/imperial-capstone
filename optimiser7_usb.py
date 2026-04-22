import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.stats import norm

# Load data
X = np.load("initial_data/function_7/initial_inputs.npy")              # shape (n, 6)
Y = np.load("initial_data/function_7/initial_outputs.npy").squeeze()   # shape (n,)

# New observed point (6D)
x_new = np.array([0.001580, 0.391010, 0.153200, 0.245729, 0.376003, 0.791463])
y_new = 1.547355474143622

# Add to dataset
X = np.append(X, [x_new], axis=0)
Y = np.append(Y, [y_new], axis=0)



print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("Y min/max:", Y.min(), Y.max())

# GP settings
alpha = 1e-3
xi = 0.01

# One length scale per dimension
kernel = RBF(
    length_scale=np.full(6, 0.2),
    length_scale_bounds=(1e-2, 10.0)
)

gp = GaussianProcessRegressor(
    kernel=kernel,
    alpha=alpha,
    normalize_y=True,
    n_restarts_optimizer=8,
    random_state=0
)

gp.fit(X, Y)

print("Learned kernel:", gp.kernel_)
print("Current best observed value:", np.max(Y))
print("Current best observed point:", X[np.argmax(Y)])

# Random candidate points in 6D
num_candidates = 100000
X_candidate = np.random.uniform(0.0, 1.0, size=(num_candidates, 6))

# GP posterior
mu, std = gp.predict(X_candidate, return_std=True)

beta = 1.5
ucb = mu + beta * std

best_idx = np.argmax(ucb)
x_next = X_candidate[best_idx]

print("Next query point:", x_next)
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