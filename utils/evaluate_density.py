import pickle
from scipy.stats import wasserstein_distance
import numpy as np

def sliced_wasserstein(X, Y, num_proj = 1024):
    dim = X.shape[1]
    ests = []
    for _ in range(num_proj):
        # sample uniformly from the unit sphere
        dir = np.random.rand(dim)
        dir /= np.linalg.norm(dir)

        # project the data
        X_proj = X @ dir
        Y_proj = Y @ dir

        # compute 1d wasserstein
        ests.append(wasserstein_distance(X_proj, Y_proj))

    return np.mean(ests)

with open("../algorithms/ERM/results/plots/MNIST_0/Z_out.pkl", "rb") as fp:
	Z_out = pickle.load(fp)

with open("out/Z_VAE.pkl", "rb") as fp:
	Z_test = pickle.load(fp)

Z_out = np.asarray(Z_out)
Z_test = np.asarray(Z_test)
print(sliced_wasserstein(Z_out, Z_test))

