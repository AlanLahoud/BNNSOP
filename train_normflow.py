import torch
import pyro
import pyro.distributions as dist
import pyro.distributions.transforms as T
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn import datasets

import sys
import data_generator
import classical_newsvendor_utils

import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler


class TrainFlowDecoupled():
    
    def __init__(self, steps, lr = 3e-3):
        X_transform = T.spline(1)
        y_transform = T.conditional_spline(1, context_dim=1)
            
        dist_base = dist.Normal(torch.zeros(1), torch.ones(1))
        self.dist_X = dist.TransformedDistribution(dist_base, [X_transform])
        self.dist_y_given_X = dist.ConditionalTransformedDistribution(dist_base, [y_transform])
        self.steps = steps
        modules = torch.nn.ModuleList([X_transform, y_transform])
        self.optimizer = torch.optim.Adam(modules.parameters(), lr)

    def train(self, X, y, X_val, y_val):
        for step in range(self.steps):
            self.optimizer.zero_grad()
            ln_p_X = self.dist_X.log_prob(X)
            ln_p_y_given_X = self.dist_y_given_X.condition(X.detach()).log_prob(y.detach())
            loss = -(ln_p_X + ln_p_y_given_X).mean()
            loss.backward()
            self.optimizer.step()
            self.dist_X.clear_cache()
            self.dist_y_given_X.clear_cache()

            if step % 500 == 0:
                with torch.no_grad():
                    ln_p_Xval = self.dist_X.log_prob(X_val)
                    ln_p_yval_given_Xval = self.dist_y_given_X.condition(X_val.detach()).log_prob(y_val.detach())
                    loss_val = -(ln_p_Xval + ln_p_yval_given_Xval).mean()
                    print(f'step: {step}, train loss: {loss.item()}, val loss: {loss_val.item()}')
        
        return self.dist_y_given_X
        
        