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
    
    def __init__(self, steps, input_size, output_size, lr = 2e-3):
        X_transform = T.spline(input_size)
        y_transform = T.conditional_spline(output_size, context_dim=input_size)
            
        dist_base = dist.Normal(torch.zeros(input_size), torch.ones(input_size))
        dist_base_out = dist.Normal(torch.zeros(output_size), torch.ones(output_size))
        
        self.dist_X = dist.TransformedDistribution(dist_base, [X_transform])
        self.dist_y_given_X = dist.ConditionalTransformedDistribution(dist_base_out, [y_transform])
        self.steps = steps
        modules = torch.nn.ModuleList([X_transform, y_transform])
        self.optimizer = torch.optim.Adam(modules.parameters(), lr)
        

    def train(self, X, y, X_val, y_val):
        for step in range(self.steps):
            self.optimizer.zero_grad()
            ln_p_X = self.dist_X.log_prob(X)
            y_given_X = self.dist_y_given_X.condition(X.detach())
            ln_p_y_given_X = y_given_X.log_prob(y.detach())
            loss = -(ln_p_X.mean() + ln_p_y_given_X.mean())
            loss.backward()
            self.optimizer.step()
            self.dist_X.clear_cache()
            self.dist_y_given_X.clear_cache()

            if step % 50 == 0:
                with torch.no_grad():
                    ln_p_Xval = self.dist_X.log_prob(X_val)
                    ln_p_yval_given_Xval = self.dist_y_given_X.condition(X_val.detach()).log_prob(y_val.detach())
                    loss_val = -(ln_p_Xval.mean() + ln_p_yval_given_Xval.mean())
                    print(f'step: {step}, train loss: {round(loss.item(), 5)}, val loss: {round(loss_val.item(), 5)}')
        
        return self.dist_y_given_X
    
    
    
class TrainFlowCombined():
    
    def __init__(self, steps, input_size, output_size, lr, OP, n_samples):
        X_transform = T.spline(input_size)
        y_transform = T.conditional_spline(output_size, context_dim=input_size)
            
        dist_base = dist.Normal(torch.zeros(input_size), torch.ones(input_size))
        dist_base_out = dist.Normal(torch.zeros(output_size), torch.ones(output_size))
        
        self.dist_X = dist.TransformedDistribution(dist_base, [X_transform])
        self.dist_y_given_X = dist.ConditionalTransformedDistribution(dist_base_out, [y_transform])
        self.steps = steps
        modules = torch.nn.ModuleList([X_transform, y_transform])
        self.optimizer = torch.optim.Adam(modules.parameters(), lr)
        self.end_loss_flow = OP.end_loss_dist
        self.n_samples = n_samples
        

    def train(self, X, y, X_val, y_val):
        for step in range(self.steps):
            #pdb.set_trace()
            self.optimizer.zero_grad()
            ln_p_X = self.dist_X.log_prob(X)
            y_preds = self.dist_y_given_X.condition(X).rsample(
                torch.Size([self.n_samples, 5000]))#.squeeze()
 
            loss = self.end_loss_flow(y_preds, y.unsqueeze(0).expand(y_preds.shape))

            loss.backward()
            self.optimizer.step()
            self.dist_X.clear_cache()
            self.dist_y_given_X.clear_cache()

            if step % 200 == 0:
                with torch.no_grad():
                    #pdb.set_trace()
                    ln_p_Xval = self.dist_X.log_prob(X_val)
                    y_val_pred = self.dist_y_given_X.condition(X_val).sample(
                        torch.Size([self.n_samples, 3000]))#.squeeze()
                    
                    loss_val = self.end_loss_flow(y_val_pred, y_val.unsqueeze(0).expand(y_val_pred.shape))
     
                    print(f'step: {step}, train loss: {round(loss.item(), 5)}, val loss: {round(loss_val.item(), 5)}')
        
        return self.dist_y_given_X
        
        