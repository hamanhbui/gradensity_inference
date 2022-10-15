import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


alpha = np.random.uniform(0, 2 * np.pi, 1000)

rad_tr = np.random.normal(3, 0.5, 1000)
x_tr = (rad_tr * np.sin(alpha), rad_tr * np.cos(alpha))
x_tr = np.array(x_tr)
x_tr = np.moveaxis(x_tr, 0, -1)

rad_te = np.random.normal(7, 0.5, 1000)
x_te = (rad_te * np.sin(alpha), rad_te * np.cos(alpha))
x_te = np.array(x_te)
x_te = np.moveaxis(x_te, 0, -1)


def gt_func(x):
    # x = [N,2]
    mu = 3
    sigma = 0.5
    x = nn.Parameter(x.detach(), requires_grad=True)
    r = torch.sqrt((x**2).sum(1, keepdims=True))
    log_prob = (-1 / 2 * ((mu - r) / sigma) ** 2).sum()
    log_prob.backward()
    return x.grad


samples = torch.tensor(x_te.astype(np.float32))
for i in range(100):
    ### X^{i+1} = X^i - \nabla_x \log p(X^i) * 0.01
    grad1 = gt_func(samples)
    samples = samples.add(grad1 * 0.01)

samples = samples.detach().numpy()

plt.scatter(x_tr[:, 0], x_tr[:, 1], color="red")
plt.scatter(x_te[:, 0], x_te[:, 1], color="blue")
plt.scatter(samples[:, 0], samples[:, 1], color="green")
plt.show()
