import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.stats import norm

# Load data
X = np.load("initial_data/function_3/initial_inputs.npy")              # shape (15, 3)
Y = np.load("initial_data/function_3/initial_outputs.npy").squeeze()   # shape (15,)

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
beta = 2.0
ucb = mu + beta * std

best_idx = np.argmax(ucb)
x_next = X_candidate[best_idx]

print("Next query point:", x_next)
print("Predicted mean there:", mu[best_idx])
print("Predicted std there:", std[best_idx])
print("UCB value there:", ucb[best_idx])

#X shape: (15, 3)
#Y shape: (15,)
#Y min/max: -0.3989255131463011 -0.034835313350078584
#Learned kernel: RBF(length_scale=0.192)
#Next query point: [0.41666667 0.41666667 0.5       ]
#Predicted mean there: -0.037238932916827686
#Predicted std there: 0.06884287989991485
#UCB value there: 0.10044682688300201