import torch
from qpth.qp import QPFunction

from mip import Model as ModelMip, xsum, minimize, maximize, \
INTEGER, BINARY, CONTINUOUS, CutType, OptimizationStatus

import numpy as np


class RiskPortOP():
    """
    Quadratic Newsvendor Stochastic Optimization Problem class.
    Init with deterministic parameters params_t and solve it for n_samples.
    """
    def __init__(self, n_samples, n_assets, min_return, Y_train, dev):
        super(RiskPortOP, self).__init__()
            
        self.dev = dev    
        self.N = n_assets
        self.M = n_samples
        
        self.R = torch.tensor(min_return) .to(self.dev)
        self.uy = torch.clip(Y_train.mean(axis=0), torch.tensor(0.01), None).to(self.dev)
        #self.uy = torch.ones_like(Y_train[0,:])
                  
        self.Q = 0.0001*torch.diag(torch.ones(self.M + self.N)).to(self.dev)
        
        self.lin = torch.hstack(( 
            (1/self.M)*torch.ones(self.M), 
            torch.zeros(self.N)
        )).to(self.dev)
        
        self.eyeM = torch.eye(self.M).to(self.dev)
        
        det_ineq = torch.hstack(( torch.zeros(self.M).to(self.dev), -self.uy )).to(self.dev)
        #det_ineq_2 = torch.hstack(( torch.zeros(self.M), self.uy, torch.tensor(0) ))
        
        #det_ineq = torch.vstack((det_ineq_1, det_ineq_2))
        
        #positive_ineq = torch.hstack( (torch.diag(-torch.ones(self.M+self.N)), 
        #                               torch.zeros(self.M+self.N).unsqueeze(0).T ))
        
        positive_ineq = torch.diag(-torch.ones(self.M+self.N)).to(self.dev)
        
        self.ineqs = torch.vstack(( det_ineq, # profit bound
                                    -det_ineq, # profit bound
                                   positive_ineq, # positive variables
                                   #-positive_ineq # bound variables
                                  )).to(self.dev)
        
        
        self.bounds = torch.hstack(( torch.tensor(-self.R).to(self.dev), # profit bound
                                    torch.tensor(1.0000001*self.R).to(self.dev), # profit bound
                                    torch.zeros(self.M + self.N).to(self.dev), # positive variables
                                    #999999.*torch.ones(self.M + self.N).to(self.dev), # bound variables
                                    torch.zeros(self.M).to(self.dev) )).to(self.dev) # max ineq
        

        
        self.e = torch.DoubleTensor().to(self.dev)
        
        
        
    def forward(self, Y_dist):
        """
        Applies the qpth solver for all batches and allows backpropagation.
        Formulation based on Priya L. Donti, Brandon Amos, J. Zico Kolter (2017).
        Note: The quadratic terms (Q) are used as auxiliar terms only to allow the backpropagation through the 
        qpth library from Amos and Kolter. 
        We will set them as a small percentage of the linear terms (Wilder, Ewing, Dilkina, Tambe, 2019)
        """
        
        batch_size, n_samples, n_assets = Y_dist.size()
        
        assert self.N == n_assets
        
        assert self.M == n_samples
              
        #import pdb
        #pdb.set_trace()

        Q = self.Q
        Q = Q.expand(batch_size, Q.size(0), Q.size(1))
        
        lin = self.lin
        lin = lin.expand(batch_size, lin.size(0))
        
        # max ineq
        unc_ineq = torch.dstack(( -self.eyeM.expand(batch_size, self.M, self.M), 
                                  -Y_dist ))
        
        ineqs = torch.unsqueeze(self.ineqs, dim=0)
        ineqs = ineqs.expand(batch_size, ineqs.shape[1], ineqs.shape[2])
                
        ineqs = torch.hstack(( ineqs, unc_ineq ))
        
        bounds = self.bounds.unsqueeze(dim=0).expand(
            batch_size, self.bounds.shape[0])
        
        argmin = QPFunction(verbose=-1)\
            (2*Q.double(), lin.double(), ineqs.double(), 
             bounds.double(), self.e, self.e).double()
        
        ustar = argmin[:, :self.M]
        zstar = argmin[:, self.M:]    
        
        if not ((torch.all(ustar >= -0.00001) and torch.all(zstar >= -0.00001))):
            print(Y_dist.min())
            print(Y_dist.max())
        
        assert torch.all(ustar >= -0.00001)
        assert torch.all(zstar >= -0.00001)
        assert (self.uy*zstar).sum()>=0.999*self.R*batch_size
        assert (self.uy*zstar).sum()<=1.001*self.R*batch_size

                    
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

    
    
    
    
    
    
    
   