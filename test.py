import numpy as np
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import matplotlib as mpl

from time import time

from sketches import gaussian, less, sparse_rademacher, srht, rrs, rrs_lev_scores

from sklearn.kernel_approximation import RBFSampler

from solvers_lr import LogisticRegression



#n = 16000
#d = 6000 
n = 1600
d = 600
lambd = 1e-5

A = np.random.randn(n,d)
u, sigma, vh = np.linalg.svd(A, full_matrices=False)
sigma = np.array([0.98**jj for jj in range(d)])
A = u @ (np.diag(sigma) @ vh)

m = 300
nnz = 0.02

xpl = 1./np.sqrt(d)*np.random.randn(d,1)
b = np.sign(A@ xpl)

A = torch.tensor(A)
b = torch.tensor(b)

lreg = LogisticRegression(A, b, lambd)

x, losses = lreg.solve_exactly(n_iter=20, eps=1e-15)

m = 50

n_iter_gd = 500
n_iter_sgd = 500
n_iter_newton = 5
n_iter_ihs = 30
n_iter_bfgs = 100

nnz = 0.005

losses_ihs = {}
times_ihs = {}

sketches = ['less_sparse', 'gaussian', 'rrs', 'srht']

_, losses_newton, times_newton = lreg.newton(n_iter=n_iter_newton)
_, losses_gd, times_gd = lreg.gd(n_iter=n_iter_gd)
_, losses_sgd, times_sgd = lreg.sgd(n_iter=n_iter_sgd, s=0.001)
_, losses_bfgs, times_bfgs = lreg.bfgs(n_iter=n_iter_bfgs)

for sketch in sketches:
    print('ihs: ', sketch)
    _, losses_, times_ = lreg.ihs(sketch_size=m, sketch=sketch, nnz=nnz, n_iter=n_iter_ihs)
    losses_ihs[sketch] = losses_
    times_ihs[sketch] = times_

print (losses_ihs)
print (times_ihs)
