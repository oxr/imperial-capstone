import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# Load data
X = np.load("initial_data/function_8/initial_inputs.npy")              # shape (n, 8)
Y = np.load("initial_data/function_8/initial_outputs.npy").squeeze()   # shape (n,)

# Add observed points
X = np.append(
    X,
    [
        [0.252147, 0.204355, 0.175316, 0.048325, 0.899664, 0.150554, 0.281117, 0.601399],
        [0.102074, 0.334362, 0.093195, 0.329480, 0.952571, 0.797051, 0.364258, 0.171929],
        [0.246050, 0.095059, 0.246398, 0.209242, 0.998847, 0.722932, 0.193943, 0.897100],
        [0.205524, 0.208265, 0.271202, 0.111435, 0.979915, 0.174137, 0.280129, 0.992042],
        [0.340517, 0.005199, 0.185794, 0.053477, 0.975096, 0.950896, 0.254186, 0.603971],
        [0.172077, 0.223987, 0.123393, 0.278475, 0.993589, 0.534333, 0.068548, 1.000000],
        [0.154056, 0.166203, 0.214579, 0.315368, 1.000000, 0.542319, 0.003479, 0.225366],
        [0.257780, 0.335906, 0.149056, 0.255531, 1.000000, 0.801790, 0.141197, 1.000000],
        [0.117842, 0.144468, 0.127133, 0.285000, 0.962514, 0.333220, 0.110359, 0.994527],   # week 11
    ],
    axis=0,
)

Y = np.append(
    Y,
    [
        9.7940105951019,
        9.7575610446754,
        9.8317962269165,
        9.7624508823461,
        9.6201698854839,
        9.8970225910435,
        9.8320189953334,
        9.769430927277,
        9.9084259223481,    # week 11
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
beta = 0.03   # very mild UCB: mostly exploitation

kernel = RBF(
    length_scale=np.array([0.8, 1.2, 0.5, 1.5, 3.0, 20.0, 0.8, 20.0]),
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
# Candidate points in constrained F8 local region
# ------------------------------------------------------------
# Preserve the structure of the current best:
# low x1/x2/x3, moderate x4, high x5, moderate x6,
# low x7, high x8.
# ------------------------------------------------------------

num_candidates = 120000

# recentred on new best (week 11): x1,x2,x6 shifted down to bracket it
lower = np.array([0.07, 0.10, 0.08, 0.22, 0.95, 0.25, 0.00, 0.85])
upper = np.array([0.17, 0.20, 0.18, 0.35, 1.00, 0.45, 0.12, 1.00])

X_candidate = np.random.uniform(lower, upper, size=(num_candidates, 8))

# Optional: force x5 and/or x8 exactly to boundary if desired
# X_candidate[:, 4] = 1.0
# X_candidate[:, 7] = 1.0

# GP posterior
mu, std = gp.predict(X_candidate, return_std=True)

# UCB acquisition
ucb = mu + beta * std

# ------------------------------------------------------------
# Avoid points too close to any previous observation
# ------------------------------------------------------------

min_dist = 0.04

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

# ConvergenceWarning: The optimal value found for dimension 7 of parameter length_scale is close to the specified upper bound 100.0. Increasing the bound and calling fit again may find a better value.
#   warnings.warn(
# Learned kernel: RBF(length_scale=[0.823, 1.18, 0.495, 1.53, 3.23, 100, 0.782, 100])
# Current best observed value: 9.7940105951019
# Current best observed point: [0.252147 0.204355 0.175316 0.048325 0.899664 0.150554 0.281117 0.601399]
# Next query point: [0.24605006 0.09505581 0.24639778 0.20924229 0.99884714 0.72293155
#  0.19394302 0.89709901]
# Predicted mean there: 9.80167743892494
# Predicted std there: 0.09048525184254538
# EI value there: 0.0349438141983468

# nceWarning: The optimal value found for dimension 7 of parameter length_scale is close to the specified upper bound 100.0. Increasing the bound and calling fit again may find a better value.
#   warnings.warn(
# Learned kernel: RBF(length_scale=[0.836, 1.21, 0.501, 1.51, 3.38, 100, 0.822, 100])
# Current best observed value: 9.8317962269165
# Current best observed point: [0.24605  0.095059 0.246398 0.209242 0.998847 0.722932 0.193943 0.8971  ]
# Next query point: [0.20552395 0.2082646  0.27120229 0.11143546 0.97991504 0.17413737
#  0.28012883 0.99204152]
# Predicted mean there: 9.832527527335865
# Predicted std there: 0.043513815869011055
# EI value there: 0.013117482298087284

# X shape: (44, 8)
# Y shape: (44,)
# Y min/max: 5.5921933895401965 9.8317962269165
# /opt/anaconda3/envs/ml-course/lib/python3.12/site-packages/sklearn/gaussian_process/kernels.py:450: ConvergenceWarning: The optimal value found for dimension 7 of parameter length_scale is close to the specified upper bound 100.0. Increasing the bound and calling fit again may find a better value.
#   warnings.warn(
# Learned kernel: RBF(length_scale=[0.827, 1.25, 0.507, 1.59, 3.36, 18.7, 0.845, 100])
# Current best observed value: 9.8317962269165
# Current best observed point: [0.24605  0.095059 0.246398 0.209242 0.998847 0.722932 0.193943 0.8971  ]
# Next query point: [0.34051747 0.00519933 0.18579414 0.05347681 0.97509625 0.95089586
#  0.25418629 0.60397129]
# Predicted mean there: 9.817541550730548
# Predicted std there: 0.07459700766034286
# EI value there: 0.019191924592293155

# X shape: (45, 8)
# Y shape: (45,)
# Y min/max: 5.5921933895401965 9.8317962269165
# Kernel: RBF(length_scale=[0.8, 1.2, 0.5, 1.5, 3, 20, 0.8, 20])
# Current best observed value: 9.8317962269165
# Current best observed point: [0.24605  0.095059 0.246398 0.209242 0.998847 0.722932 0.193943 0.8971  ]
# Next query point: [0.17207693 0.22398737 0.12339337 0.27847515 0.99358863 0.53433251
#  0.06854845 1.        ]
# Distance from best: 0.3223779306815221
# Predicted mean there: 10.051775206586587
# Predicted std there: 0.09468076235950948
# EI value there: 0.22030264618750306

# X shape: (46, 8)
# Y shape: (46,)
# Y min/max: 5.5921933895401965 9.8970225910435
# Kernel: RBF(length_scale=[0.8, 1.2, 0.5, 1.5, 3, 20, 0.8, 20])
# Current best observed value: 9.8970225910435
# Current best observed point: [0.172077 0.223987 0.123393 0.278475 0.993589 0.534333 0.068548 1.      ]
# Next query point: [0.15405612 0.16620304 0.21457866 0.31536758 1.         0.54231936
#  0.00347851 0.22536603]
# Distance from best: 0.785962042786844
# Predicted mean there: 9.945287696009038
# Predicted std there: 0.07834389724102617
# EI value there: 0.061137722331419335

# X shape: (47, 8)
# Y shape: (47,)
# Y min/max: 5.5921933895401965 9.8970225910435
# Kernel: RBF(length_scale=[0.8, 1.2, 0.5, 1.5, 3, 20, 0.8, 20])
# Current best observed value: 9.8970225910435
# Current best observed point: [0.172077 0.223987 0.123393 0.278475 0.993589 0.534333 0.068548 1.      ]
# Next query point: [0.25777957 0.33590615 0.14905555 0.25553094 1.         0.80179034
#  0.14119792 1.        ]
# Distance from best: 0.3129031224403656
# Predicted mean there: 9.934327657136988
# Predicted std there: 0.04555106524845431
# EI value there: 0.04259990815260608

# X shape: (48, 8)
# Y shape: (48,)
# Y min/max: 5.5921933895401965 9.8970225910435
# Kernel: RBF(length_scale=[0.8, 1.2, 0.5, 1.5, 3, 20, 0.8, 20])
# Current best observed value: 9.8970225910435
# Current best observed point: [0.172077 0.223987 0.123393 0.278475 0.993589 0.534333 0.068548 1.      ]
# Next query point: [0.12671698 0.1611935  0.08198043 0.25727738 0.9758364  0.41275557
#  0.09777374 0.98351002]
# Distance from best: 0.15616396754599707
# Distance to nearest observed: 0.15616396754599707
# Predicted mean there: 9.978677224426612
# Predicted std there: 0.020190207978903207
# UCB value there: 9.97928293066598