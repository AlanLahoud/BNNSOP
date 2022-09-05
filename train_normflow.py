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
import classical_newsvendor_utils as cnu

import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

import pdb

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
                    print(f'step: {step}, train loss: {round(loss.item(), 5)}, val loss: {round(loss_val.item(), 5)}')
        
        return self.dist_y_given_X
    
    
    
class TrainFlowCombined():
    
    def __init__(self, steps, lr, sell_price, cost_price, n_samples):
        X_transform = T.spline(1)
        y_transform = T.conditional_spline(1, context_dim=1)
            
        dist_base = dist.Normal(torch.zeros(1), torch.ones(1))
        self.dist_X = dist.TransformedDistribution(dist_base, [X_transform])
        self.dist_y_given_X = dist.ConditionalTransformedDistribution(dist_base, [y_transform])
        self.steps = steps
        modules = torch.nn.ModuleList([X_transform, y_transform])
        self.optimizer = torch.optim.Adam(modules.parameters(), lr)
        self.sell_price = sell_price
        self.cost_price = cost_price
        self.n_samples = n_samples
        
    def end_loss_flow(self, y_pred_dist, y_true):
        z_pred = cnu.get_argmins_from_dist(self.sell_price, self.cost_price, y_pred_dist)
        end_loss = -cnu.profit_sum(z_pred, y_true, self.sell_price, self.cost_price)
        return end_loss

    def train(self, X, y, X_val, y_val):
        for step in range(self.steps):
            pdb.set_trace()
            self.optimizer.zero_grad()
            ln_p_X = self.dist_X.log_prob(X)
            y_pred = self.dist_y_given_X.condition(X.detach()).sample(
                torch.Size([self.n_samples, 5000])).squeeze()
            
            loss = self.end_loss_flow(y_pred, y)
            #ln_p_y_given_X = y_pred.log_prob(y.detach())
            #loss = -(ln_p_X + ln_p_y_given_X).mean()
            loss.backward()
            self.optimizer.step()
            self.dist_X.clear_cache()
            self.dist_y_given_X.clear_cache()

            if step % 500 == 0:
                with torch.no_grad():
                    ln_p_Xval = self.dist_X.log_prob(X_val)
                    y_val_pred = self.dist_y_given_X.condition(X_val.detach()).sample(
                        torch.Size([self.n_samples,])).squeeze()
                    loss_val = self.end_loss_flow(y_val_pred, y_val)
                    #ln_p_yval_given_Xval = y_val_pred.log_prob(y_val.detach())
                    #loss_val = -(ln_p_Xval + ln_p_yval_given_Xval).mean()
                    print(f'step: {step}, train loss: {round(loss.item(), 5)}, val loss: {round(loss_val.item(), 5)}')
        
        return self.dist_y_given_X
        
        