import math
import pickle

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics

# prerequisites
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image


def idx2onehot(idx, n):
	assert torch.max(idx).item() < n
	if idx.dim() == 1:
		idx = idx.unsqueeze(1)
	onehot = torch.zeros(idx.size(0), n).to(idx.device)
	onehot.scatter_(1, idx, 1)
	return onehot


class Score_Func(nn.Module):
	def __init__(self):
		super().__init__()
		self.fc = nn.Sequential(nn.Linear(4, 2))

	def forward(self, x, c):
		c = idx2onehot(c, n = 2)
		x = torch.cat((x, c), dim=-1)
		score = self.fc(x)
		return score


def score_matching(score_net, samples, labels, n_particles=1):
	dup_samples = samples.unsqueeze(0).expand(n_particles, *samples.shape).contiguous().view(-1, *samples.shape[1:])
	dup_samples.requires_grad_(True)

	grad1 = score_net(dup_samples, labels)
	loss1 = torch.norm(grad1, dim=-1) ** 2 / 2.0

	loss2 = torch.zeros(dup_samples.shape[0], device=dup_samples.device)
	for i in range(dup_samples.shape[1]):
		grad = torch.autograd.grad(grad1[:, i].sum(), dup_samples, create_graph=True)[0][:, i]
		loss2 += grad

	loss = loss1 + loss2
	return loss.mean()


def create_dataset():
	# Create dataset
	alpha_0 = np.random.uniform(2 / 3 * np.pi, 0.85 * np.pi, 100)
	alpha_1 = np.random.uniform(1.2 * np.pi, 3 / 2 * np.pi, 100)

	rad_tr = np.random.normal(3, 0.5, 100)
	x_tr_0 = (rad_tr * np.sin(alpha_0), rad_tr * np.cos(alpha_0))
	y_tr_0 = np.zeros(100)
	x_tr_1 = (rad_tr * np.sin(alpha_1), rad_tr * np.cos(alpha_1))
	y_tr_1 = np.ones(100)

	x_tr = np.concatenate((x_tr_0, x_tr_1), axis=1)
	x_tr = np.moveaxis(x_tr, 0, -1)
	y_tr = np.concatenate((y_tr_0, y_tr_1))

	rad_te = np.random.normal(10, 0.5, 100)
	noise_te = np.random.normal(0, 0.15, 100)
	x_te = (rad_te * noise_te, rad_te)
	x_te = np.array(x_te)
	x_te = np.moveaxis(x_te, 0, -1)

	y_te = []
	for x in x_te:
		if x[0] > 0:
			y_te.append(0)
		else:
			y_te.append(1)

	y_te = np.array(y_te)

	return x_tr, y_tr, x_te, y_te


def loss_function(recon_x, x, mu, log_var):
	BCE = F.mse_loss(recon_x, x.view(-1, 2), reduction="sum")
	KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
	return BCE + KLD


def main():
	# Create dataset
	x_tr, y_tr, x_te, y_te = create_dataset()

	# Train model on training dataset
	# LogisticRegression
	model = LogisticRegression()
	model.fit(x_tr, y_tr)

	# Estimate density function
	score_func = Score_Func()
	density_optimizer = torch.optim.Adam(score_func.parameters(), lr = 1e-3)

	density_losses = []
	iters = []
	score_func.cuda()
	score_func.train()
	for ite in range(5000):
		samples = torch.tensor(x_tr.astype(np.float32)).cuda()
		lables = torch.tensor(y_tr.astype(np.int64)).cuda()
		density_loss = score_matching(score_func, samples.detach(), lables)
		density_losses.append(density_loss)
		iters.append(ite)

		density_optimizer.zero_grad()
		density_loss.backward()
		density_optimizer.step()

	# Fast-adaptation on testing dataset
	x_te_adapted = torch.tensor(x_te.astype(np.float32))
	grads = []
	score_func.eval()

	# Create List of list
	for i in range(100):
	    grads.append([])

	score_func.cpu()
	with torch.no_grad():
		# Estimated density
		for i in range(100):
			y_hat = model.predict(x_te_adapted)
			lables = torch.tensor(y_te.astype(np.int64))
			grad1 = score_func(x_te_adapted, lables)
			if i % 10 == 0:
				for i in range(len(x_te_adapted)):
					grads[i].append(x_te_adapted[i] + grad1[i] * 0.01)
			x_te_adapted = x_te_adapted.add(grad1 * 0.01)

	x_te_adapted = x_te_adapted.detach().numpy()

	# Print accuracies
	y_hat = model.predict(x_te)
	print(sklearn.metrics.accuracy_score(y_hat, y_te))
	y_hat = model.predict(x_te_adapted)
	print(sklearn.metrics.accuracy_score(y_hat, y_te))

	# Prepare for visualization
	h = 0.02  # point in the mesh [x_min, m_max]x[y_min, y_max].

	x_min, x_max = x_tr[:, 0].min() - 9, x_tr[:, 0].max() + 9
	y_min, y_max = x_tr[:, 1].min() - 5, x_tr[:, 1].max() + 12
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

	# Predictions to obtain the classification results
	Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

	# Plotting
	fig, ax = plt.subplots()
	cdict = {0: "violet", 1: "yellow"}
	for g in np.unique(y_te):
		i = np.where(y_te == g)
		ax.scatter(x_te[i, 0], x_te[i, 1], label=g, c=cdict[g])
		ax.scatter(x_te_adapted[i, 0], x_te_adapted[i, 1], label=g, c=cdict[g])

	for i in range(0, len(grads)):
		for j in range(1, 10):
			ax.annotate(
				"",
				xy=grads[i][j],
				xytext=grads[i][j - 1],
				arrowprops={"arrowstyle": "->", "color": "b"},
				va="center",
				ha="center",
			)

	plt.gca().add_patch(plt.Circle((0, 0), 3, color="green", fill=False))
	plt.plot([-8, 8], [0, 0], linestyle="--", color="black")
	plt.plot([0, 0], [-8, 8], linestyle="--", color="green")

	plt.contourf(xx, yy, Z, alpha=0.4)
	plt.scatter(x_tr[:, 0], x_tr[:, 1], c=y_tr, alpha=0.8)

	plt.xlabel("Dim-1")
	plt.ylabel("Dim-2")

	plt.savefig("demo.jpg")


if __name__ == "__main__":
	main()
