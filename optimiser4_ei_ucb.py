import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# Load data
X = np.load("initial_data/function_4/initial_inputs.npy")              # shape (n, 4)
Y = np.load("initial_data/function_4/initial_outputs.npy").squeeze()   # shape (n,)

# New observed points
x_new1 = np.array([0.460348, 0.439865, 0.399919, 0.434560])
y_new1 = -0.42194022156229805

x_new2 = np.array([0.444690, 0.438661, 0.335267, 0.439198])
y_new2 = -0.17606848295603994

x_new = np.array([0.404162, 0.423633, 0.372091, 0.440380])
y_new = 0.364702696138949

# Add new points to dataset
X = np.append(
    X,
    [
        x_new1,
        x_new2,
        [0.404162, 0.423633, 0.372091, 0.440380],
        [0.450559, 0.422909, 0.304881, 0.424339],
        [0.316603, 0.493962, 0.385730, 0.490159],
        [0.383641, 0.376630, 0.366120, 0.428964],
        [0.383642, 0.376637, 0.366121, 0.428965],
        [0.389930, 0.404481, 0.367830, 0.410122],
        [0.380393, 0.363049, 0.408871, 0.409785]   # week 11
    ],
    axis=0,
)

Y = np.append(
    Y,
    [
        -0.42194022156229805,
        -0.17606848295603994,
        0.364702696138949,
        -0.7802788857897229,
        -3.0949732106943943,
        0.43954640909770193,
        0.43945330978694885,
        0.4462220200654383,
        0.568628271769811      # week 11
    ],
    axis=0,
)

print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("Y min/max:", Y.min(), Y.max())

# GP model
alpha = 1e-6
beta = 0.03   # small beta = more exploitative UCB

kernel = RBF(length_scale=0.25, length_scale_bounds="fixed")

gp = GaussianProcessRegressor(
    kernel=kernel,
    alpha=alpha,
    normalize_y=True,
)

gp.fit(X, Y)

print("Learned kernel:", gp.kernel_)

# Candidate points in local 4D box
num_candidates = 50000

# recentred on new best [0.3804, 0.3630, 0.4089, 0.4098] (week 11)
lower = np.array([0.35, 0.33, 0.38, 0.38])
upper = np.array([0.41, 0.39, 0.44, 0.44])

X_candidate = np.random.uniform(lower, upper, size=(num_candidates, 4))

# GP posterior
mu, std = gp.predict(X_candidate, return_std=True)

# UCB acquisition for maximisation
ucb = mu + beta * std

# Mask already observed / nearly observed points
tol = 1e-6

for x_obs in X:
    duplicate_mask = np.linalg.norm(X_candidate - x_obs, axis=1) < tol
    ucb[duplicate_mask] = -np.inf

# Avoid another near-duplicate of the current best point
best_x = X[np.argmax(Y)]

too_close_to_best = np.linalg.norm(X_candidate - best_x, axis=1) < 0.015
ucb[too_close_to_best] = -np.inf

# Next query point
best_idx = np.argmax(ucb)
x_next = X_candidate[best_idx]


print("Best observed value:", np.max(Y))
print("Best observed point:", X[np.argmax(Y)])
print("Next query point:", x_next)
print("Predicted mean there:", mu[best_idx])
print("Predicted std there:", std[best_idx])
print("UCB value there:", ucb[best_idx])
print("Distance from best:", np.linalg.norm(x_next - X[np.argmax(Y)]))


#X shape: (30, 4)
#Y shape: (30,)
#Y min/max: -32.625660215962455 -4.025542281908162
#Learned kernel: RBF(length_scale=0.481)
#Best observed value: -4.025542281908162
#Next query point: [0.45810222 0.434231   0.37867107 0.44444325]
#Predicted mean there: -1.0449277189001123
#Predicted std there: 1.2189833430974326
#EI value there: 2.973579507714067

# X shape: (31, 4)
# Y shape: (31,)
# Y min/max: -32.625660215962455 0.42194022156229805
# Learned kernel: RBF(length_scale=0.491)
# Best observed value: 0.42194022156229805
# Next query point: [0.44468993 0.43866107 0.33526684 0.4391981 ]
# Predicted mean there: -0.062271532450729694
# Predicted std there: 0.7974381177171157
# EI value there: 0.13023850144030744

# X shape: (32, 4)
# Y shape: (32,)
# Y min/max: -32.625660215962455 0.42194022156229805
# Learned kernel: RBF(length_scale=0.506)
# Best observed value: 0.42194022156229805
# Next query point: [0.40416254 0.42363341 0.37209122 0.44038032]
# Predicted mean there: -0.08986385980774259
# Predicted std there: 0.5875136656707663
# EI value there: 0.06029639885200397

# -- correction to y values

# X shape: (33, 4)
# Y shape: (33,)
# Y min/max: -32.625660215962455 0.364702696138949
# Learned kernel: RBF(length_scale=0.521)
# Best observed value: 0.364702696138949
# Next query point: [0.45055919 0.42290845 0.30488046 0.42433895]
# Predicted mean there: -0.4600338175342422
# Predicted std there: 0.6290405694363804
# EI value there: 0.027033730482462617

# skipped the last result which was very close to the previous one, including the bad result
# this one is more exploitative 

# X shape: (34, 4)
# Y shape: (34,)
# Y min/max: -32.625660215962455 0.364702696138949
# Learned kernel: RBF(length_scale=0.393)
# Best observed value: 0.364702696138949
# Next query point: [0.31660319 0.49396222 0.38572978 0.49015876]
# Predicted mean there: 1.1126820405080622
# Predicted std there: 0.6234939060278195
# EI value there: 0.7829834368863064

# change to UCB

# X shape: (35, 4)
# Y shape: (35,)
# Y min/max: -32.625660215962455 0.364702696138949
# Learned kernel: RBF(length_scale=0.25)
# Best observed value: 0.364702696138949
# Best observed point: [0.404162 0.423633 0.372091 0.44038 ]
# Next query point: [0.38364162 0.3766373  0.36612072 0.42896472]
# Predicted mean there: 0.6156845637221089
# Predicted std there: 0.5041982652465545
# UCB value there: 0.6661043902467644
# Distance from best: 0.052873757167937886

# X shape: (37, 4)
# Y shape: (37,)
# Y min/max: -32.625660215962455 0.43954640909770193
# Learned kernel: RBF(length_scale=0.25)
# Best observed value: 0.43954640909770193
# Best observed point: [0.383641 0.37663  0.36612  0.428964]
# Next query point: [0.38992993 0.40448083 0.36783017 0.41012225]
# Predicted mean there: 0.5252423671521438
# Predicted std there: 0.5036926302904292
# UCB value there: 0.5756116301811867
# Distance from best: 0.034251352383331865

# X shape: (38, 4)
# Y shape: (38,)
# Y min/max: -32.625660215962455 0.4462220200654383
# Learned kernel: RBF(length_scale=0.25)
# Best observed value: 0.4462220200654383
# Best observed point: [0.38993  0.404481 0.36783  0.410122]
# Next query point: [0.39089429 0.39649494 0.3612061  0.4326174 ]
# Predicted mean there: 0.502766546580375
# Predicted std there: 0.077735181628235
# UCB value there: 0.5050986020292221
# Distance from best: 0.02479164961614386