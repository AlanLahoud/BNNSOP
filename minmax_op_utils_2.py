import torch
from qpth.qp import QPFunction

from mip import Model as ModelMip, xsum, minimize, maximize, \
INTEGER, BINARY, CONTINUOUS, CutType, OptimizationStatus

import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer

import numpy as np


class RiskPortOP():

    def __init__(self, n_samples, n_assets, min_return, Y_train, dev):
        super(RiskPortOP, self).__init__()
            
        self.dev = dev    
        self.N = n_assets
        self.M = n_samples
        
        self.R = torch.tensor(min_return)#.to(self.dev)
        self.uy = torch.clip(Y_train.mean(axis=0), 0.1, None)#.to(self.dev)

        
        
    def forward(self, Y_dist):
        
        batch_size, n_samples, n_assets = Y_dist.size()
        
        assert self.N == n_assets        
        assert self.M == n_samples              

        u = cp.Variable(self.M)
        z = cp.Variable(self.N)

        Y_dist_param = cp.Parameter((self.M, self.N))

        constraints = [u >= 0.00, z>=0.00, Y_dist_param @ z + u >=0.00, 
                       cp.sum(cp.multiply(z,self.uy)) >= self.R]


        objective = cp.Minimize((1/self.M) * cp.sum(u)) 


        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp()

        cvxpylayer = CvxpyLayer(problem, parameters=[Y_dist_param], variables=[u, z])

        Y_dist_param = Y_dist

        ustar, zstar = cvxpylayer(Y_dist_param)
        
        import pdb
        if not torch.all(zstar >= -0.01):
            pdb.set_trace()
    
        assert torch.all(ustar >= -0.1)
        assert torch.all(zstar >= -0.01)
        assert (self.uy*zstar).sum()>=0.999*self.R*batch_size

                    
        return ustar, zstar
    
    
    def risk_loss_dataset(self, Y_dist, zstar_pred):        
        loss_portfolio = -(Y_dist*zstar_pred.unsqueeze(1)).sum(2)
        u = loss_portfolio.squeeze()    
        loss_risk = (torch.max(u, torch.zeros_like(u)))/Y_dist.shape[1]
        return loss_risk

    def calc_f_dataset(self, Y_dist_pred, Y_dist, optm=False):
        Y_dist_pred = Y_dist_pred.permute((1, 0, 2))
        if optm:
            zstar_pred = self.forward_true(Y_dist_pred.cpu().detach().numpy())
            zstar_pred = torch.tensor(np.array(zstar_pred)).to(self.dev)
        else:
            _, zstar_pred = self.forward(Y_dist_pred)
        loss_risk = self.risk_loss_dataset(Y_dist, zstar_pred)
        return loss_risk
    
    def cost_fn(self, y_pred, y, optm=False):
        f = self.calc_f_dataset(y_pred, y, optm)
        f_total = torch.mean(f)
        return f_total

    def end_loss(self, y_pred, y, optm=False):
        y_pred = y_pred.unsqueeze(0)
        y = y.unsqueeze(1)
        f_total = self.cost_fn(y_pred, y, optm)
        return f_total

    def end_loss_dist(self, y_pred, y, optm=False):
        if y.dim()==2:
            y = y.unsqueeze(1)
        f_total = self.cost_fn(y_pred, y, optm)
        return f_total
    
    
    def min_true_sample(self, y):
    
        n_samples = y.shape[0]
        n_assets = y.shape[1]
        
        uy = self.uy.cpu().detach().numpy()
        R = self.R.cpu().detach().numpy()
        
        assert self.N == n_assets
           
        m = ModelMip("cvar")
        m.verbose = 0
        z = ([m.add_var(var_type=CONTINUOUS, name=f'z_{i}') for i in range(0, n_assets)])
        u = ([m.add_var(var_type=CONTINUOUS, name=f'u_{i}') for i in range(0, n_samples)])

        m.objective = minimize((1/n_samples)*xsum(u[i] for i in range(0, n_samples)))

        for i in range(0, n_assets):
            m += z[i] >= 0

        for i in range(0, n_samples):
            m += u[i] >= 0

        for i in range(0, n_samples):
            m += xsum(z[j]*y[i][j] for j in range(0, n_assets)) + u[i] >= 0

        m += xsum(-z[i]*uy[i] for i in range(0, n_assets)) <= -R

        m.optimize()
        f_opt = m.objective_value
        argmins = []
        for v in m.vars:
            argmins.append(v.x)
           
        zstar = argmins[:n_assets]
        ustar = argmins[n_assets:]

        return ustar, zstar
    
    
    def forward_true(self, Y_dist):
        zstar = np.zeros_like(Y_dist[:,0,:])
        for i in range(0, Y_dist.shape[0]):
            _ , zstar[i,:] = self.min_true_sample(Y_dist[i,:,:])       
        return zstar

    
    
    
    
    
    
    
   