import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import torch
from sklearn.base import clone

class GP(): 
    """
    Class that uses sklearn library to create a GP model.  
    """
    def __init__(self, length_scale, length_scale_bounds, 
                 alpha_noise, white_noise, n_restarts_optimizer):
        self.ls = length_scale
        self.lsbs = length_scale_bounds
        self.alp = alpha_noise
        self.nro = n_restarts_optimizer
        self.wn = white_noise
        self.gp_reg = GaussianProcessRegressor(
                        kernel=self.define_kernel(), 
                        alpha=self.alp, 
                        n_restarts_optimizer=self.nro)
        self.M = 512
    
    def define_kernel(self):
        """
        Defines the RBF kernel + White noise Kernel
        """
        return RBF(self.ls, self.lsbs) + WhiteKernel(
            noise_level=self.wn, 
            noise_level_bounds=(self.wn/100, self.wn*100))
    
    def update_n_samples(self, n_samples):
        """
        Update number of samples when predicting samples given input
        """
        self.M = n_samples
    
    def gp_fit(self, X, y):
        self.gp_reg.fit(X, y)
        return self.gp_reg
        
    def forward_dist(self, X_test, aleat_bool):
        """
        Predict M samples given GP trained model and input X
        """
        try:
            X_test = X_test.to('cpu')
        except:
            pass
        y_aux = self.gp_reg.sample_y(X_test,n_samples=self.M)
        if y_aux.ndim==2:
            y_aux = y_aux.T
            y_dist = torch.tensor(np.expand_dims(y_aux, -1))
        else:
            y_dist = torch.permute(torch.tensor(y_aux), (2,0,1))
        return y_dist