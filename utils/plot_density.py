import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import seaborn as sns
from scipy.stats import wasserstein_distance


# with open("../algorithms/VAE/results/plots/Rotated_75_MNIST_0/tr_entropies.pkl", "rb") as fp:
#     train_NLL = pickle.load(fp)

# with open("../algorithms/VAE/results/plots/Rotated_75_MNIST_0/te_entropies.pkl", "rb") as fp:
#     test_NLL = pickle.load(fp)

# with open("../algorithms/VAE/results/plots/Rotated_75_MNIST_0/adapt_entropies.pkl", "rb") as fp:
#     adapted_NLL = pickle.load(fp)

with open("../algorithms/VAE/results/plots/Rotated_75_MNIST_0/tr_nlls.pkl", "rb") as fp:
    train_NLL = pickle.load(fp)

with open("../algorithms/VAE/results/plots/Rotated_75_MNIST_0/te_nlls.pkl", "rb") as fp:
    test_NLL = pickle.load(fp)

with open("../algorithms/VAE/results/plots/Rotated_75_MNIST_0/adapt_nlls.pkl", "rb") as fp:
    adapted_NLL = pickle.load(fp)

plt.figure(figsize=(20, 10))
plt.xlabel("Entropy")
plt.hist(train_NLL, label="train", density=True, bins=int(180 / 5))
plt.hist(test_NLL, label="test", density=True, bins=int(180 / 5))
plt.hist(adapted_NLL, label="adapt", density=True, bins=int(180 / 5))
plt.legend()
plt.savefig("out/out.png")

# # seaborn histogram
# sns.kdeplot(train_NLL, label="train")
# sns.kdeplot(test_NLL, label="test")
# sns.kdeplot(adapted_NLL, label="adapted")

# plt.legend()
# plt.savefig("out/predictive_entropy.png")
