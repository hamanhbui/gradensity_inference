import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import sklearn.metrics


class Score_Func(nn.Module):
	def __init__(self):
		super().__init__()
		self.fc = nn.Linear(2, 2)
 
	def forward(self, x):
		score = self.fc(x)
		return score


def score_matching(score_net, samples, n_particles=1):
	dup_samples = samples.unsqueeze(0).expand(n_particles, *samples.shape).contiguous().view(-1, *samples.shape[1:])
	dup_samples.requires_grad_(True)

	grad1 = score_net(dup_samples)
	loss1 = (torch.norm(grad1, dim=-1) ** 2 / 2.)
 
	loss2 = torch.zeros(dup_samples.shape[0], device=dup_samples.device)
	for i in range(2):
		grad = torch.autograd.grad(grad1[:, i].sum(), dup_samples, create_graph=True)[0][:, i]
		loss2 += grad
 
	loss = loss1 + loss2
	return loss.mean()


def gt_func(x):
	# x = [N,2]
	mu = 3
	sigma = 0.5
	x = nn.Parameter(x.detach(), requires_grad=True)
	r = torch.sqrt((x**2).sum(1,keepdims=True))
	log_prob = (-1/2 * ((mu-r)/sigma)**2).sum()
	log_prob.backward()
	return x.grad


def create_dataset():
	# Create dataset
	alpha_0 = np.random.uniform(2/3 * np.pi, 0.85 * np.pi, 100)
	alpha_1 = np.random.uniform(1.2 * np.pi, 3/2 * np.pi, 100)
	
	rad_tr = np.random.normal(3, 0.5, 100)
	x_tr_0 = (rad_tr * np.sin(alpha_0), rad_tr * np.cos(alpha_0))
	y_tr_0 = np.zeros(100)
	x_tr_1 = (rad_tr * np.sin(alpha_1), rad_tr * np.cos(alpha_1))
	y_tr_1 = np.ones(100)

	x_tr = np.concatenate((x_tr_0, x_tr_1), axis = 1)
	x_tr = np.moveaxis(x_tr, 0, -1)
	y_tr = np.concatenate((y_tr_0, y_tr_1))

	rad_te = np.random.normal(-10, 0.5, 100)
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


def main():
	# Create dataset
	x_tr, y_tr, x_te, y_te = create_dataset()

	# Train model on training dataset
	# LogisticRegression
	model = LogisticRegression()
	# # Gaussian Naive Bayes
	# model = GaussianNB()
	model.fit(x_tr, y_tr)

	# Estimate density function
	score_func = Score_Func()
	density_optimizer = torch.optim.Adam(score_func.parameters(), lr = 1e-3)
	
	density_losses = []
	iters = []
	for ite in range(10000):
		samples = torch.tensor(x_tr.astype(np.float32))
		density_loss = score_matching(score_func, samples.detach())
		density_losses.append(density_loss)
		iters.append(ite)
	
		density_optimizer.zero_grad()
		density_loss.backward()
		density_optimizer.step()

	# Fast-adaptation on testing dataset
	x_te_adapted = torch.tensor(x_te.astype(np.float32))
	
	# # Known density
	# for i in range(100):
	# 	### X^{i+1} = X^i - \nabla_x \log p(X^i) * 0.01
	# 	grad1 = gt_func(x_te_adapted)
	# 	x_te_adapted = x_te_adapted.add(grad1 * 0.01)

	# Estimated density
	for i in range(100):
		grad1 = score_func(x_te_adapted)
		x_te_adapted = x_te_adapted.add(grad1 * 0.01)
	
	x_te_adapted = x_te_adapted.detach().numpy()

	# Print accuracies
	y_hat = model.predict(x_te)
	print(sklearn.metrics.accuracy_score(y_hat, y_te))
	y_hat = model.predict(x_te_adapted)
	print(sklearn.metrics.accuracy_score(y_hat, y_te))

	# Prepare for visualization
	h = .02 # point in the mesh [x_min, m_max]x[y_min, y_max].    

	x_min, x_max = x_tr[:, 0].min() - 9, x_tr[:, 0].max() + 9
	y_min, y_max = x_tr[:, 1].min() - 9, x_tr[:, 1].max() + 9
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

	# Predictions to obtain the classification results
	Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

	# Plotting
	fig, ax = plt.subplots()
	cdict = {0: 'violet', 1: 'yellow'}
	for g in np.unique(y_te):
		i = np.where(y_te == g)
		ax.scatter(x_te[i, 0], x_te[i, 1], label=g, c = cdict[g])
		ax.scatter(x_te_adapted[i, 0], x_te_adapted[i, 1], label=g, c = cdict[g])

	plt.gca().add_patch(plt.Circle((0, 0), 3, color='green', fill=False))
	plt.plot([-8, 8], [0, 0], linestyle="--", color = "black")
	plt.plot([0, 0], [-8, 8], linestyle="--", color = "green")

	plt.contourf(xx, yy, Z, alpha=0.4)
	plt.scatter(x_tr[:, 0], x_tr[:, 1], c=y_tr, alpha=0.8)

	plt.xlabel("Dim-1")
	plt.ylabel("Dim-2")

	plt.savefig("demo.jpg")


if __name__ == "__main__":
	main()