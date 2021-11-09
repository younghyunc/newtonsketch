import torch
from torch import cholesky, cholesky_solve
import numpy as np

from sketches import gaussian, srht, less, sparse_rademacher, rrs

from utils import average 
from time import time 



SKETCHES = {'gaussian': gaussian, 'srht': srht, 'less_sparse': sparse_rademacher, 'less': less, 'rrs': rrs} 


def direct_method(data_matrix, target):

    if target.ndim == 1:
        target = target.reshape((-1,1))

    start = time()
    x_opt = torch.lstsq(target, data_matrix)[0][:data_matrix.shape[1]]
    baseline_time = time() - start 
    return x_opt, baseline_time 



class Solver:

    def __init__(self, data_matrix, target, x_opt, reg_param=1e-2):

        self.A = data_matrix
        if target.ndim == 1:
            target = target.reshape((-1,1))
        self.b = target 
        (self.n, self.d), self.c = self.A.shape, self.b.shape[1]
        self.x_opt = x_opt
        self.reg_param = reg_param
        if self.x_opt.ndim == 1:
            self.x_opt = self.x_opt.reshape((-1,1))


    def compute_error(self, x):
        
        return torch.mean( torch.log( 1 + torch.exp(-self.b * (self.A @ x)) )) + self.reg_param/2 * (x**2).sum()
        



class IHS(Solver):


    def __init__(self, A, b, x_opt, sketch='gaussian'):

        Solver.__init__(self, A, b, x_opt)

        self.sketch = sketch 
        self.sketch_fn = SKETCHES[sketch]


    def compute_step_size(self, m, q, x, p, g):

        if not self.line_search:
            gamma = m / (m-self.d)
            return q/(gamma*(gamma-1+q))
        else:
            return (p*g).sum() / (p*(self.A.T @ (self.A @ p))).sum()


    def ihs_iteration(self, x, m, q):

        start = time()

        g = self.A.T @ (self.A @ x -self.b)
        p = torch.zeros(self.d, self.c).to(self.A.device)
        for _ in range(q):
            sa = self.sketch_fn(self.A, m, nnz=self.nnz)
            U = cholesky(sa.T @ sa)
            p += 1./q * cholesky_solve(g, U)

        mu = self.compute_step_size(m, q, x, p, g)
        x = x - mu * p
        return x, time() - start


    #@average
    def solve(self, m, q=1, line_search=False, n_iterations=100, n_trials=1, nnz=None, error_threshold=1e-6):
        if nnz is None:
            self.nnz = self.d 
        else:
            self.nnz = nnz 

        self.line_search = line_search 

        x = 1./np.sqrt(self.d) * torch.randn(self.d, self.c)
        x = x.to(self.A.device)
        errors = [self.compute_error(x)]
        times = []
        iteration = 0

        success = False
        for _ in range(n_iterations):
            x, time_ = self.ihs_iteration(x, m, q)
            error = self.compute_error(x)
            errors.append(error)
            times.append(time_)
            #errors.append(self.compute_error(x))

            if error != 1 and error < error_threshold:
                success = True
                break

        if success == False:
            times = [999999]

        print ("errors: ", errors)

        cv_rate = (errors[-1]/errors[0])**(1./n_iterations)
        return x, torch.Tensor(errors)/errors[0], cv_rate, np.cumsum(times)
