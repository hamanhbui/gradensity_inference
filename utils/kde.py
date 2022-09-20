import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
import pickle

from sklearn.manifold import TSNE

with open("algorithms/ERM/results/plots/MNIST_0/Z_out.pkl", "rb") as fp:
	Z_out = pickle.load(fp)

with open("algorithms/ERM/results/plots/Rotated_75_MNIST_0/Z_test.pkl", "rb") as fp:
	Z_test = pickle.load(fp)

data = np.asarray(Z_out)

# use grid search cross-validation to optimize the bandwidth
params = {"bandwidth": np.logspace(-1, 1, 20)}
grid = GridSearchCV(KernelDensity(), params)
grid.fit(data)

print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))

# use the best estimator to compute the kernel density estimate
kde = grid.best_estimator_

tsne_model = TSNE(n_components=2, init="pca")
Z_2d = tsne_model.fit_transform(data)

plt.scatter(Z_2d[:, 0], Z_2d[:, 1], marker=".")

test_data = kde.sample(1000)

tsne_model = TSNE(n_components=2, init="pca")
Z_2d = tsne_model.fit_transform(test_data)

plt.scatter(Z_2d[:, 0], Z_2d[:, 1], marker=".")
plt.savefig("test.jpg")
