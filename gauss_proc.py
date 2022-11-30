import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import torch

class GP():    
    def __init__(self, length_scale, length_scale_bounds, 
                 alpha_noise, n_restarts_optimizer):
        self.ls = length_scale
        self.lsbs = length_scale_bounds
        self.alp = alpha_noise
        self.nro = n_restarts_optimizer
        self.gp_reg = GaussianProcessRegressor(
                        kernel=self.define_kernel(), 
                        alpha=self.alp, 
                        n_restarts_optimizer=self.nro)
        self.M = 512
    
    def define_kernel(self):
        return RBF(self.ls, self.lsbs)
    
    def update_n_samples(self, n_samples):
        self.M = n_samples
    
    def gp_fit(self, X, y):
        self.gp_reg.fit(X, y)
        return self.gp_reg
        
    def forward_dist(self, X_test, aleat_bool):
        y_aux = self.gp_reg.sample_y(X_test,n_samples=self.M)
        if y_aux.ndim==2:
            y_aux = y_aux.T
            y_dist = torch.tensor(np.expand_dims(y_aux, -1))
        else:
            y_dist = torch.permute(torch.tensor(y_aux), (2,0,1))
        return y_dist