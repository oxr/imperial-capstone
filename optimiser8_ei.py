import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.stats import norm

# Load data
X = np.load("initial_data/function_8/initial_inputs.npy")              # shape (n, 8)
Y = np.load("initial_data/function_8/initial_outputs.npy").squeeze()   # shape (n,)

# New observed point (8D)
x_new = np.array([0.252147, 0.204355, 0.175316, 0.048325, 0.899664, 0.150554, 0.281117, 0.601399])
y_new = 9.7940105951019

# Add to dataset
X = np.append(X, [x_new], axis=0)
Y = np.append(Y, [y_new], axis=0)


print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("Y min/max:", Y.min(), Y.max())

# GP settings
alpha = 1e-3
xi = 0.01

# One learned length scale per dimension
kernel = RBF(
    length_scale=np.full(8, 0.2),
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

# Random candidate points in 8D
num_candidates = 200000
X_candidate = np.random.uniform(0.0, 1.0, size=(num_candidates, 8))

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

print("Next query point:", x_next)
print("Predicted mean there:", mu[best_idx])
print("Predicted std there:", std[best_idx])
print("EI value there:", ei[best_idx])

# (base) Mac-mini:initial_data oxr$ /opt/anaconda3/envs/ml-course/bin/python /Users/oxr/src/python/aiml/capstone2/initial_data/optimiser8_ei.py
#X shape: (40, 8)
#Y shape: (40,)
#Y min/max: 5.5921933895401965 9.598482002566342
#/opt/anaconda3/envs/ml-course/lib/python3.12/site-packages/sklearn/gaussian_process/kernels.py:450: ConvergenceWarning: The optimal value found for dimension 5 of parameter length_scale is close to the specified upper bound 10.0. Increasing the bound and calling fit again may find a better value.
#  warnings.warn(
#/opt/anaconda3/envs/ml-course/lib/python3.12/site-packages/sklearn/gaussian_process/kernels.py:450: ConvergenceWarning: The optimal value found for dimension 7 of parameter length_scale is close to the specified upper bound 10.0. Increasing the bound and calling fit again may find a better value.
#  warnings.warn(
#Learned kernel: RBF(length_scale=[0.794, 1.09, 0.482, 1.81, 2.98, 10, 0.745, 10])
#Current best observed value: 9.598482002566342
#Current best observed point: [0.05644741 0.06595555 0.02292868 0.03878647 0.40393544 0.80105533 0.48830701 0.89308498]
#Next query point: [0.24004686 0.23324658 0.24934576 0.10286222 0.94199113 0.59802266 0.20544369 0.41759764]
#Predicted mean there: 9.994939672332913
#Predicted std there: 0.1684102738963742
#EI value there: 0.38708388566214286

# (base) Mac-mini:imperial-capstone oxr$ /opt/anaconda3/envs/ml-course/bin/python /Users/oxr/src/python/imperial-capstone/optimiser8_ei.py
# X shape: (41, 8)
# Y shape: (41,)
# Y min/max: 5.5921933895401965 9.7940105951019
# /opt/anaconda3/envs/ml-course/lib/python3.12/site-packages/sklearn/gaussian_process/kernels.py:450: ConvergenceWarning: The optimal value found for dimension 5 of parameter length_scale is close to the specified upper bound 10.0. Increasing the bound and calling fit again may find a better value.
#   warnings.warn(/opt/anaconda3/envs/ml-course/lib/python3.12/site-packages/sklearn/gaussian_process/kernels.py:450: ConvergenceWarning: The optimal value found for dimension 7 of parameter length_scale is close to the specified upper bound 10.0. Increasing the bound and calling fit again may find a better value.
#   warnings.warn(
# Learned kernel: RBF(length_scale=[0.798, 1.14, 0.488, 1.76, 3.01, 10, 0.761, 10])
# Current best observed value: 9.7940105951019
# Current best observed point: [0.252147 0.204355 0.175316 0.048325 0.899664 0.150554 0.281117 0.601399]
# Next query point: [0.10207362 0.33436154 0.09319465 0.32947952 0.95257145 0.79705184 0.36425834 0.17192856]
# Predicted mean there: 9.819600321523987
# Predicted std there: 0.15069945143138502
# EI value there: 0.06823665523241504