import sys
import torch
import numpy as np
import math
from tqdm import tqdm
import copy
 


class TrainDecoupled():
    
    def __init__(self, bnn, model, opt, loss_data, K, aleat_bool, training_loader, validation_loader, dev):
        self.model = model
        self.opt = opt
        self.loss_data = loss_data
        self.K = K
        self.aleat_bool = aleat_bool
        self.training_loader = training_loader
        self.validation_loader = validation_loader
        self.bnn = bnn
        self.dev = dev
        
        self.logsqrttwopi = torch.log(
            torch.sqrt(2*torch.tensor(math.pi)))
      
    
    def train_one_epoch(self):
        data_running_loss = 0.
        kl_running_loss = 0.

        n = len(self.training_loader.dataset)
        n_batches = len(self.training_loader)
                
        for i, data in enumerate(self.training_loader):

            x_batch, y_batch = data
            x_batch = x_batch.to(self.dev)
            y_batch = y_batch.to(self.dev)
            
            self.opt.zero_grad()
 
            y_preds, rho_preds = self.model(x_batch)

            if self.bnn:
                #pdb.set_trace()
                #y_preds = y_preds.mean(axis=0)
                y_batch = y_batch.unsqueeze(0).expand(y_preds.shape)
                
                if self.aleat_bool:
                    loss_data_ = self.loss_data(y_preds, y_batch)*torch.exp(-rho_preds) + rho_preds
                else:
                    loss_data_ = self.loss_data(y_preds, y_batch)

                #loss_data_ = loss_data_.min(axis=0).values.mean(axis=0) + loss_data_.min(axis=1).values.mean(axis=0)
                loss_data_ = loss_data_.mean(axis=1) #through batch
                loss_data_ = loss_data_.mean(axis=0) #through stochastic weights
                kl_loss_ = self.K*self.model.kl_divergence_NN()/n_batches

            else:
                
                if self.aleat_bool:
                    loss_data_ = self.loss_data(y_preds, y_batch)*torch.exp(-rho_preds) + rho_preds
                else:
                    loss_data_ = self.loss_data(y_preds, y_batch)
                
                loss_data_ = loss_data_.mean(axis=0) #through batch
                kl_loss_ = torch.tensor(0)
                
            loss_data_ = loss_data_.mean(axis=-1) #through output dimension

            total_loss = loss_data_ + kl_loss_
            total_loss.backward()

            self.opt.step()

            data_running_loss += loss_data_.item()
            kl_running_loss += kl_loss_.item()

        loss_data = data_running_loss/n_batches
        kl = kl_running_loss

        return loss_data, kl
    
    
    def train(self, EPOCHS=150):
        epoch_number = 0
 
        avg_best_vloss = np.inf
        best_model = copy.deepcopy(self.model)
    
        for epoch in tqdm(range(EPOCHS)):
            

            self.model.train(True)
            avg_loss_data_loss, avg_kl_loss = self.train_one_epoch()
            avg_loss = avg_loss_data_loss + avg_kl_loss

            self.model.train(False)
            loss_data_running_loss_v = 0.0

            n = len(self.validation_loader.dataset)
            n_batches = len(self.validation_loader)
            
            if self.bnn:
                kl_loss_ = self.K*self.model.kl_divergence_NN()
            else:
                kl_loss_ = torch.tensor(0)

            for i, vdata in enumerate(self.validation_loader):

                x_val_batch, y_val_batch = vdata
                
                x_val_batch = x_val_batch.to(self.dev)
                y_val_batch = y_val_batch.to(self.dev)

                y_val_preds, rho_val_preds = self.model(x_val_batch)
                
                if self.bnn:
                    y_val_batch = y_val_batch.unsqueeze(0).expand(y_val_preds.shape)
                    
                    if self.aleat_bool:
                        loss_data_ = self.loss_data(y_val_preds, y_val_batch)*torch.exp(-rho_val_preds) + rho_val_preds
                    else:
                        loss_data_ = self.loss_data(y_val_preds, y_val_batch)
                                     
                    loss_data_ = loss_data_.mean(axis=1) #through batch
                    loss_data_ = loss_data_.mean(axis=0) #through stochastic weights
                    #loss_data_ = loss_data_.min(axis=0).values.mean(axis=0) + loss_data_.min(axis=1).values.mean(axis=0)
                    
                else:
                    if self.aleat_bool:
                        loss_data_ = self.loss_data(y_val_preds, y_val_batch)*torch.exp(-rho_val_preds) + rho_val_preds
                    else:
                        loss_data_ = self.loss_data(y_val_preds, y_val_batch)
                        
                    loss_data_ = loss_data_.mean() #through batch
                
                loss_data_ = loss_data_.mean(axis=-1) #through output dimension

                loss_data_running_loss_v += loss_data_

                
            avg_vloss_data = (loss_data_running_loss_v/n_batches).item()
            avg_vklloss = (kl_loss_).item()

            avg_vloss = avg_vloss_data + avg_vklloss

            if epoch_number == 0 or (epoch_number+1)%10 == 0:           
                print('------------------EPOCH {}------------------'.format(
                    epoch_number + 1))

                print('DATA LOSS \t train {} valid {}'.format(
                    round(avg_loss_data_loss, 3), round(avg_vloss_data, 3)))
                print('KL LOSS \t train {} valid {}'.format(
                    round(avg_kl_loss/(self.K+0.000001), 2), round(avg_vklloss/(self.K+0.000001), 2)))
                print('ELBO LOSS \t train {} valid {}'.format(
                    round(avg_loss, 2), round(avg_vloss, 2)))
    
            if  avg_vloss < avg_best_vloss:
                avg_best_vloss = avg_vloss
                best_model=copy.deepcopy(self.model)
            
            epoch_number += 1
            
        return best_model 
            

            
class TrainCombined():
    
    def __init__(self, bnn, model, opt, aleat_bool, training_loader, validation_loader, OP, dev):
        self.model = model
        self.opt = opt
        self.aleat_bool = aleat_bool
        self.training_loader = training_loader
        self.validation_loader = validation_loader
        self.bnn = bnn
        self.end_loss = OP.end_loss
        self.end_loss_dist = OP.end_loss_dist
        self.dev = dev
       
   
    def train_one_epoch(self):

        n = len(self.training_loader.dataset)
        n_batches = len(self.training_loader)
        
        end_total_loss = 0

        
        for i, data in enumerate(self.training_loader):

            x_batch, y_batch = data
            x_batch = x_batch.to(self.dev)
            y_batch = y_batch.to(self.dev)
            
            self.opt.zero_grad()
 
            y_preds, rho_preds = self.model(x_batch)

            if self.aleat_bool:
                y_preds = y_preds + torch.sqrt(
                    torch.exp(rho_preds))*torch.randn(
                    y_preds.size(), device = self.dev)
    
            if self.bnn:
                y_batch = y_batch.unsqueeze(0).expand(y_preds.shape)
                total_loss = self.end_loss_dist(y_preds, y_batch)

            else:
                total_loss = self.end_loss(y_preds, y_batch)
                

            total_loss.backward()
            self.opt.step()

            end_total_loss += total_loss.item()

        end_total_loss = end_total_loss/n_batches

        return end_total_loss
    
    
    
    def train(self, EPOCHS=150):
        epoch_number = 0
        
        avg_best_vloss = np.inf
        best_model = copy.deepcopy(self.model)
        
        for epoch in tqdm(range(EPOCHS)):

            self.model.train(True)
            avg_loss = self.train_one_epoch()

            self.model.train(False)

            n = len(self.validation_loader.dataset)
            n_batches = len(self.validation_loader)
            
            total_running_loss_v = 0.0
            
            for i, vdata in enumerate(self.validation_loader):

                x_val_batch, y_val_batch = vdata
                
                x_val_batch = x_val_batch.to(self.dev)
                y_val_batch = y_val_batch.to(self.dev)
                      
                y_val_preds, rho_val_preds = self.model(x_val_batch)

                if self.aleat_bool:
                    y_val_preds = y_val_preds + torch.sqrt(
                        torch.exp(rho_val_preds))*torch.randn(
                        y_val_preds.size(), device = self.dev)

                if self.bnn:
                    y_val_batch = y_val_batch.unsqueeze(0).expand(y_val_preds.shape)
                    total_loss_v = self.end_loss_dist(y_val_preds, y_val_batch)

                else:
                    total_loss_v = self.end_loss(y_val_preds, y_val_batch)
                
                total_running_loss_v += total_loss_v.item()

            if epoch_number == 0 or (epoch_number+1)%10 == 0: 
                print('------------------EPOCH {}------------------'.format(
                    epoch_number + 1))
                print('END LOSS \t train {} valid {}'.format(
                    round(avg_loss, 3), round(total_running_loss_v/n_batches, 3)))

            if  total_running_loss_v/n_batches < avg_best_vloss:
                avg_best_vloss = total_running_loss_v/n_batches
                best_model=copy.deepcopy(self.model)
                
            epoch_number += 1
            
        return best_model