import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# Load data
X = np.load("initial_data/function_5/initial_inputs.npy")
Y = np.load("initial_data/function_5/initial_outputs.npy").squeeze()

# New observed point
x_new = np.array([0.301974, 0.854760, 0.955726, 0.956521])
y_new = 2131.370600010672

# Add to dataset
X = np.append(X, [x_new], axis=0)
Y = np.append(Y, [y_new], axis=0)

print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("Y min/max:", Y.min(), Y.max())

# GP model
alpha = 1e-4
beta = 1.2

kernel = RBF(length_scale=0.3, length_scale_bounds=(1e-2, 2.0))
gp = GaussianProcessRegressor(
    kernel=kernel,
    alpha=alpha,
    normalize_y=True,
    n_restarts_optimizer=5
)

gp.fit(X, Y)

print("Learned kernel:", gp.kernel_)

# Random candidate points in 4D
num_candidates = 50000
X_candidate = np.random.uniform(0.0, 1.0, size=(num_candidates, 4))

# GP posterior
mu, std = gp.predict(X_candidate, return_std=True)

# UCB acquisition
ucb = mu + beta * std

# Next query point
best_idx = np.argmax(ucb)
x_next = X_candidate[best_idx]

print("Best observed value:", np.max(Y))
print("Next query point:", x_next)
print("Predicted mean there:", mu[best_idx])
print("Predicted std there:", std[best_idx])
print("UCB value there:", ucb[best_idx])

#X shape: (20, 4)
#Y shape: (20,)
#Y min/max: 0.1129397953712203 1088.8596181962705
#Learned kernel: RBF(length_scale=0.269)
#Best observed value: 1088.8596181962705
#Predicted std there: 96.26810412921864
#UCB value there: 1206.1755546884594

# (base) Mac-mini:imperial-capstone oxr$ /opt/anaconda3/envs/ml-course/bin/python /Users/oxr/src/python/imperial-capstone/optimiser5_ucb.py
# X shape: (21, 4)
# Y shape: (21,)
# Y min/max: 0.1129397953712203 2131.370600010672
# Learned kernel: RBF(length_scale=0.01)
# Best observed value: 2131.370600010672
# Next query point: [0.09265463 0.87067141 0.64019798 0.85580181]
# Predicted mean there: 248.17524802715803
# Predicted std there: 484.977335989895
# UCB value there: 830.148051215032