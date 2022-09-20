import argparse
import os
import pickle

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE


def plot_TNSE(X_2d_tr, Z_sample, filename = "Z_class_tSNE1.png"):
	colors = ["red", "green", "blue", "black", "brown", "grey", "orange", "yellow", "pink", "cyan", "magenta"]
	plt.figure(figsize=(16, 16))
	plt.scatter(X_2d_tr[:, 0], X_2d_tr[:, 1], marker=".")
	plt.scatter(Z_sample[:, 0], Z_sample[:, 1], marker=".")

	plt.savefig(filename)


# def unique(list1):
#     unique_list = []
#     for x in list1:
#         if x not in unique_list:
#             unique_list.append(x)
#     return unique_list


def tsne_plot(Z_out, Z_sample):

	tsne_model = TSNE(n_components=2, init="pca")
	Z_2d = tsne_model.fit_transform(Z_out)
	Z_sample = tsne_model.fit_transform(Z_sample)
	

	plot_TNSE(Z_2d, Z_sample)


if __name__ == "__main__":
	# parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	# parser.add_argument("--plotdir", help="Path to configuration file")
	# bash_args = parser.parse_args()
	# dir_name = bash_args.plotdir

	with open("../algorithms/ERM/results/plots/MNIST_0/Z_out.pkl", "rb") as fp:
		Z_out = pickle.load(fp)

	with open("Z_out.pkl", "rb") as fp:
		Z_sample = pickle.load(fp)
	# with open(dir_name + "Y_out.pkl", "rb") as fp:
	#     Y_out = pickle.load(fp)

	# for i in range(len(Y_out)):
	#     Y_out[i] = Y_out[i][0]

	tsne_plot(Z_out, Z_sample)