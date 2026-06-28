import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.stats import norm

# Load data
X0 = np.load("initial_data/function_3/initial_inputs.npy")
Y0 = np.load("initial_data/function_3/initial_outputs.npy").squeeze()



x_new = [0.416667, 0.541667, 0.541667]
y_new = -0.014143360975926783


X = np.append(X0, [[0.375000, 0.416667, 0.458333], 
                   [0.416667, 0.250000, 0.500000], 
                   [0.416667, 0.541667, 0.541667],
                   [0.541667, 0.708333, 0.583333],
                   [0.458333, 0.791667, 0.000000],
                   [0.750000, 0.791667, 0.000001],
                   [0.396598, 0.652098, 0.482640],
                   [0.396598, 0.652099, 0.482640],
                   [0.459106, 0.600553, 0.479508],
                   [0.388758, 0.620817, 0.509439],  # week 10
                   [0.371860, 0.602664, 0.524613]   # week 11
                   ], axis=0)
Y = np.append(Y0, [-0.026529173868188035,
                   -0.04248359577869199,
                   -0.014143360975926783,
                   -0.04713365704600794,
                   -0.12011662189405359,
                   -0.11512601211975466,
                   -0.016146506049970505,
                   -0.007517478976212062,
                   -0.02189000347551078,
                   -0.00592324039282754,  # week 10 — new best
                   -0.008922005878200353   # week 11
                   ], axis=0)




print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("Y min/max:", Y.min(), Y.max())

# GP model
alpha = 1e-2
xi = 0.0
kernel = RBF(length_scale=0.18, length_scale_bounds="fixed")
gp = GaussianProcessRegressor(
    kernel=kernel,
    alpha=alpha,
    normalize_y=True,
    n_restarts_optimizer=5
)

gp.fit(X, Y)

print("Learned kernel:", gp.kernel_)

num_candidates = 80000

# New best at [0.389, 0.621, 0.509] — recentre search
lower = np.array([0.36, 0.59, 0.47])
upper = np.array([0.42, 0.65, 0.54])

X_candidate = np.random.uniform(lower, upper, size=(num_candidates, 3))

mu, std = gp.predict(X_candidate, return_std=True)

beta = 0.03
acquisition = mu + beta * std

min_dist = 0.015
too_close = np.min(
    np.linalg.norm(X_candidate[:, None, :] - X[None, :, :], axis=2),
    axis=1
) < min_dist

acquisition[too_close] = -np.inf

best_idx = np.argmax(acquisition)
x_next = X_candidate[best_idx]

print("Best observed value:", np.max(Y))
print("Best observed point:", X[np.argmax(Y)])
print("Next query point:", x_next)
print("Distance to nearest observed:", np.min(np.linalg.norm(X - x_next, axis=1)))
print("Predicted mean there:", mu[best_idx])
print("Predicted std there:", std[best_idx])
print("Acquisition value there:", acquisition[best_idx])


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection="3d")

sc = ax.scatter(
    X0[:, 0], X0[:, 1], X0[:, 2],
    c=Y0, cmap="viridis", s=50, label="Original data"
)

ax.scatter(
    x_new[0], x_new[1], x_new[2],
    c="red", s=120, marker="x", label="New point"
)

ax.scatter(
    x_next[0], x_next[1], x_next[2],
    c="white", edgecolors="black", s=120, label="Next query"
)

ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("x3")
ax.set_title("3D input space with Y encoded by colour")
ax.legend()

cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Y value")

plt.tight_layout()
plt.show()


#X shape: (15, 3)
#Y shape: (15,)
#Y min/max: -0.3989255131463011 -0.034835313350078584
#Learned kernel: RBF(length_scale=0.192)
#Best observed value: -0.034835313350078584
#Next query point: [0.375      0.41666667 0.45833333]
#Predicted mean there: -0.027007692778697606
#Predicted std there: 0.060746079927152294
#EI value there: 0.023163484770095156

# X shape: (17, 3)
# Y shape: (17,)
# Y min/max: -0.3989255131463011 0.026529173868188035
# Learned kernel: RBF(length_scale=0.225)
# Best observed value: 0.026529173868188035
# Next query point: [0.41666667 0.25       0.5       ]
# Predicted mean there: 0.0045226375235165595
# Predicted std there: 0.053173478009149566
# EI value there: 0.00894085158797539


# X shape: (18, 3)
# Y shape: (18,)
# Y min/max: -0.3989255131463011 0.026529173868188035
# Learned kernel: RBF(length_scale=0.209)
# Best observed value: 0.026529173868188035
# Next query point: [0.41666667 0.54166667 0.54166667]
# Predicted mean there: 0.005455548498303403
# Predicted std there: 0.044500257302222734
# EI value there: 0.00637671306612132

# X shape: (19, 3)
# Y shape: (19,)
# Y min/max: -0.3989255131463011 -0.014143360975926783
# Learned kernel: RBF(length_scale=0.215)
# Best observed value: -0.014143360975926783
# Next query point: [0.54166667 0.70833333 0.58333333]
# Predicted mean there: -0.02845579013603105
# Predicted std there: 0.05350268738704398
# EI value there: 0.011354861071200045

# X shape: (20, 3)
# Y shape: (20,)
# Y min/max: -0.3989255131463011 -0.014143360975926783
# Learned kernel: RBF(length_scale=0.216)
# Best observed value: -0.014143360975926783
# Next query point: [0.45833333 0.79166667 0.        ]
# Predicted mean there: -0.026189391886737808
# Predicted std there: 0.046431312224569886
# EI value there: 0.012726808173309874

#removed accidental double points
#added protection from choosing the same point twice

# X shape: (20, 3)
# Y shape: (20,)
# Y min/max: -0.3989255131463011 -0.014143360975926783
# Learned kernel: RBF(length_scale=0.187)
# Best observed value: -0.014143360975926783
# Next query point: [0.75       0.79166667 0.        ]
# Predicted mean there: -0.03731347469066933
# Predicted std there: 0.05859814219801256
# EI value there: 0.013253126034695509

# X shape: (21, 3)
# Y shape: (21,)
# Y min/max: -0.3989255131463011 -0.014143360975926783
# Learned kernel: RBF(length_scale=0.18)
# Best observed value: -0.014143360975926783
# Next query point: [0.39659842 0.65209869 0.48264074]
# Predicted mean there: -0.01902931527678349
# Predicted std there: 0.0349409492259186
# EI value there: 0.011632506955866224

# X shape: (23, 3)
# Y shape: (23,)
# Y min/max: -0.3989255131463011 -0.007517478976212062
# Learned kernel: RBF(length_scale=0.18)
# Best observed value: -0.007517478976212062
# Next query point: [0.45910558 0.60055269 0.47950757]
# Predicted mean there: -0.00949762775052286
# Predicted std there: 0.01598303848105826
# EI value there: 0.005435107573753003

# submitted point: [0.459106,0.600553,0.479508]

# X shape: (24, 3)
# Y shape: (24,)
# Y min/max: -0.3989255131463011 -0.007517478976212062
# Learned kernel: RBF(length_scale=0.18)
# Best observed value: -0.007517478976212062
# Best observed point: [0.396598 0.652099 0.48264 ]
# Next query point: [0.38875838 0.62081713 0.50943854]
# Distance to nearest observed: 0.04192987246484065
# Predicted mean there: -0.009693886345263844
# Predicted std there: 0.008490693003010016
# Acquisition value there: -0.009439165555173544
# X shape: (25, 3)
# Y shape: (25,)
# Best observed value: -0.00592324039282754  (week 10 — new best)
# Best observed point: [0.388758 0.620817 0.509439]
# Next query point: [0.371860, 0.602664, 0.524613]
# Predicted mean there: -0.00673932
# Predicted std there: 0.009147
