import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# Load data
X = np.load("initial_data/function_5/initial_inputs.npy")
Y = np.load("initial_data/function_5/initial_outputs.npy").squeeze()


# Add to dataset
X = np.append(X, [[0.301974, 0.854760, 0.955726, 0.956521], 
                  [0.092654, 0.870671, 0.640197, 0.855801], 
                  [0.301305, 0.852492, 0.998576, 0.983722],
                  [0.352490, 0.851471, 0.969132, 0.984943],
                  [0.356337, 0.830255, 0.993780, 0.962275],
                  [0.322292, 0.869417, 1.000000, 1.000000],
                  [0.336087, 0.927681, 1.000000, 1.000000],
                  [0.349437, 0.973910, 1.000000, 1.000000],
                  [0.395159, 0.999737, 0.999359, 0.999566]   # week 11
                  ], axis=0)
Y = np.append(Y, [2131.370600010672, 
                  467.59271295184953, 
                  2784.5912711217393,
                  2511.1486924861874,
                  2406.275844120267,
                  3111.0526131448732,
                  3631.236887736192,
                  4156.230674508436,
                  4517.0730486328275], axis=0)   # week 11

print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("Y min/max:", Y.min(), Y.max())

# GP model — more exploitative
alpha = 1e-6
beta = 0.15

kernel = RBF(length_scale=0.15, length_scale_bounds="fixed")

gp = GaussianProcessRegressor(
    kernel=kernel,
    alpha=alpha,
    normalize_y=True
)

gp.fit(X, Y)

print("Kernel:", gp.kernel_)

# Candidate points in 4D: mostly local around current best
best_x = X[np.argmax(Y)]
best_y = np.max(Y)

num_global = 10000
num_local = 50000

X_global = np.random.uniform(0.0, 1.0, size=(num_global, 4))

local_scale = 0.04
X_local = best_x + np.random.normal(0.0, local_scale, size=(num_local, 4))
X_local = np.clip(X_local, 0.0, 1.0)

X_candidate = np.vstack([X_global, X_local])

# GP posterior
mu, std = gp.predict(X_candidate, return_std=True)

# UCB acquisition
ucb = mu + beta * std

# Avoid re-querying observed points
tol = 1e-6
for x_obs in X:
    duplicate_mask = np.linalg.norm(X_candidate - x_obs, axis=1) < tol
    ucb[duplicate_mask] = -np.inf

# Next query point
best_idx = np.argmax(ucb)
x_next = X_candidate[best_idx]

print("Best observed value:", best_y)
print("Best observed point:", best_x)
print("Next query point:", x_next)
print("Distance from best:", np.linalg.norm(x_next - best_x))
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

# X shape: (22, 4)
# Y shape: (22,)
# Y min/max: 0.1129397953712203 2131.370600010672
# Learned kernel: RBF(length_scale=0.128)
# Best observed value: 2131.370600010672
# Next query point: [0.30130492 0.85249204 0.99857601 0.98372184]
# Predicted mean there: 2044.0648764126063
# Predicted std there: 160.37216291221424
# UCB value there: 2236.5114719072635

# X shape: (23, 4)
# Y shape: (23,)
# Y min/max: 0.1129397953712203 2784.5912711217393
# Learned kernel: RBF(length_scale=0.0937)
# Best observed value: 2784.5912711217393
# Next query point: [0.35249017 0.8514707  0.96913212 0.98494299]
# Predicted mean there: 2136.534471806267
# Predicted std there: 360.23889033285286
# UCB value there: 2568.8211402056904

# X shape: (24, 4)
# Y shape: (24,)
# Y min/max: 0.1129397953712203 2784.5912711217393
# Learned kernel: RBF(length_scale=0.133)
# Best observed value: 2784.5912711217393
# Next query point: [0.3563372  0.83025464 0.99377513 0.96227453]
# Predicted mean there: 2563.5646716841566
# Predicted std there: 227.59920607143414
# UCB value there: 2836.6837189698776

# X shape: (25, 4)
# Y shape: (25,)
# Y min/max: 0.1129397953712203 2784.5912711217393
# Kernel: RBF(length_scale=0.15)
# Best observed value: 2784.5912711217393
# Best observed point: [0.301305 0.852492 0.998576 0.983722]
# Next query point: [0.32229183 0.86941651 1.         1.        ]
# Distance from best: 0.031525977608706714
# Predicted mean there: 2957.173143242388
# Predicted std there: 88.42843884985172
# UCB value there: 2970.4374090698657

# X shape: (26, 4)
# Y shape: (26,)
# Y min/max: 0.1129397953712203 3111.0526131448732
# Kernel: RBF(length_scale=0.15)
# Best observed value: 3111.0526131448732
# Best observed point: [0.322292 0.869417 1.       1.      ]
# Next query point: [0.33608714 0.92768125 1.         1.        ]
# Distance from best: 0.05987510830025609
# Predicted mean there: 3316.3349264057015
# Predicted std there: 197.92748949272385
# UCB value there: 3346.0240498296102

# X shape: (27, 4)
# Y shape: (27,)
# Y min/max: 0.1129397953712203 3631.236887736192
# Kernel: RBF(length_scale=0.15)
# Best observed value: 3631.236887736192
# Best observed point: [0.336087 0.927681 1.       1.      ]
# Next query point: [0.34943673 0.97391044 1.         1.        ]
# Distance from best: 0.04811836211013955
# Predicted mean there: 3760.358251891067
# Predicted std there: 139.30815172441626
# UCB value there: 3781.2544746497297

# X shape: (28, 4)
# Y shape: (28,)
# Y min/max: 0.1129397953712203 4156.230674508436
# Kernel: RBF(length_scale=0.15)
# Best observed value: 4156.230674508436
# Best observed point: [0.349437 0.97391  1.       1.      ]
# Next query point: [0.37528748 1.         1.         1.        ]
# Distance from best: 0.036727856757026306
# Predicted mean there: 4438.371190343015
# Predicted std there: 130.0499480162873
# UCB value there: 4457.878682545458