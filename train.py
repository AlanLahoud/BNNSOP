import sys
import torch
import numpy as np
import math
from tqdm import tqdm
import copy

class TrainDecoupled():
    """
    Class to help on training process using the 
    Decoupled Learning Approach (see paper).
    """
    
    def __init__(self, bnn, model, opt, loss_data, K, 
                 aleat_bool, training_loader, 
                 validation_loader, dev, explr=0.99):
        self.model = model # Neural network (ANN or BNN)
        self.opt = opt
        self.loss_data = loss_data # NLL loss
        self.K = K # Useful only for BNN
        self.aleat_bool = aleat_bool # True if modeling aleatoric uncer
        self.training_loader = training_loader
        self.validation_loader = validation_loader
        self.bnn = bnn # True if BNN, False if ANN
        self.dev = dev
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            opt, gamma=explr)
        
        self.logsqrttwopi = torch.log(
            torch.sqrt(2*torch.tensor(math.pi)))
      
    
    def train_one_epoch(self, start_rho):
        """
        Update ANN or BNN weights with Decoupled Learning approach 
        for one epoch. 
        """
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
                y_batch = y_batch.unsqueeze(0).expand(y_preds.shape)
                if self.aleat_bool:
                    loss_data_ = self.loss_data(y_preds, y_batch)
                    if start_rho:
                        loss_data_ = loss_data_*torch.exp(-rho_preds) + rho_preds
                        
                else:
                    loss_data_ = self.loss_data(y_preds, y_batch)
                loss_data_ = loss_data_.mean(axis=1) #through batch
                loss_data_ = loss_data_.mean(axis=0) #through stochastic weights
                kl_loss_ = self.K*self.model.kl_divergence_NN()/n_batches

            else:
                if self.aleat_bool:
                    loss_data_ = self.loss_data(
                        y_preds, y_batch)*torch.exp(-rho_preds) + rho_preds
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
    
    
    def train(self, EPOCHS=350):
        """
        Update ANN or BNN weights with Decoupled Learning approach 
        for EPOCHS epochs.
        """
        epoch_number = 0
 
        avg_best_vloss = np.inf
        best_model = copy.deepcopy(self.model)
        
        es_count = 0
        es_count_trs = 40
    
        for epoch in tqdm(range(EPOCHS)):
            
            self.model.train(True)
            
            start_rho = False
            
            if epoch>50:
                start_rho = True
                
            avg_loss_data_loss, avg_kl_loss = self.train_one_epoch(start_rho)
            avg_loss = avg_loss_data_loss + avg_kl_loss

            self.model.train(False)
            loss_data_running_loss_v = 0.0

            n = len(self.validation_loader.dataset)
            n_batches = len(self.validation_loader)
            
            if self.bnn:
                kl_loss_ = self.K*self.model.kl_divergence_NN()
            else:
                kl_loss_ = torch.tensor(0)

            # Validation metrics
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
                    
                else:
                    if self.aleat_bool:
                        loss_data_ = self.loss_data(y_val_preds, y_val_batch)*torch.exp(-rho_val_preds) + rho_val_preds
                    else:
                        loss_data_ = self.loss_data(y_val_preds, y_val_batch)                        
                    loss_data_ = loss_data_.mean() #through batch
                
                loss_data_ = loss_data_.mean(axis=-1) #through output dimension
                loss_data_running_loss_v += loss_data_.detach()
              
            avg_vloss_data = (loss_data_running_loss_v/n_batches).item()
            avg_vklloss = (kl_loss_).item()

            avg_vloss = avg_vloss_data + avg_vklloss

            if epoch_number == 0 or (epoch_number+1)%1 == 0:           
                print('------------------EPOCH {}------------------'.format(
                    epoch_number + 1))

                print('DATA LOSS \t train {} valid {}'.format(
                    round(avg_loss_data_loss, 3), round(avg_vloss_data, 3)))
                print('KL LOSS \t train {} valid {}'.format(
                    round(avg_kl_loss/(self.K+0.000001), 2), round(avg_vklloss/(self.K+0.000001), 2)))
                print('ELBO LOSS \t train {} valid {}'.format(
                    round(avg_loss, 2), round(avg_vloss, 2)))
    
            if  avg_vloss < avg_best_vloss:
                es_count = 0
                avg_best_vloss = avg_vloss
                best_model=copy.deepcopy(self.model)
            else:
                es_count = es_count+1
            
            epoch_number += 1
            self.scheduler.step()
            
            if es_count>es_count_trs:
                print('EARLY STOP')
                break
            
        return best_model 
            

            
class TrainCombined():
    """
    Class to help on training process using the 
    Combined Learning Approach (see paper).
    """
    def __init__(self, bnn, model, opt, K, aleat_bool, 
                 training_loader, scaler, validation_loader, 
                 OP, dev, explr=0.99, OP_simple=None):
        self.model = model # Neural network (ANN or BNN)
        self.opt = opt
        self.K = K # Useful only for BNN
        self.aleat_bool = aleat_bool # True if allow noise modeling
        self.training_loader = training_loader
        self.scaler = scaler # To denormalize to solve the OP in the training process
        self.scaler_mean = torch.tensor(self.scaler.mean_, device=dev)
        self.scaler_std = torch.tensor(self.scaler.scale_, device=dev)
        self.validation_loader = validation_loader
        self.bnn = bnn # True if BNN, False if ANN
        self.end_loss = OP.end_loss # OP cost function
        self.end_loss_dist = OP.end_loss_dist # OP expect cost function
        self.dev = dev
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            opt, gamma=explr)
        
        self.n_samples = OP.n_samples
        self.OP = OP
        self.OP_simple=OP_simple
       

    def inverse_transform(self, inp):
        """
        Denormalize the data to solve the OP
        """
        return inp*self.scaler_std + self.scaler_mean
       
    def train_one_epoch(self, start_aleat=True):
        """
        Update ANN or BNN weights with Combined Learning approach 
        for one epoch. 
        """
        n = len(self.training_loader.dataset)
        n_batches = len(self.training_loader)

        end_total_loss = 0.
        kl_running_loss = 0.
        
        for i, data in enumerate(self.training_loader):
            x_batch, y_batch = data
            x_batch = x_batch.to(self.dev)
            y_batch = y_batch.to(self.dev)
            
            self.opt.zero_grad()
 
            y_preds, rho_preds = self.model(x_batch)       

            if self.aleat_bool and start_aleat:
                y_preds = y_preds + torch.sqrt(
                    torch.exp(rho_preds))*torch.randn(
                    y_preds.size(), device = self.dev)
    
            y_preds = self.inverse_transform(y_preds)
            y_batch = self.inverse_transform(y_batch)
            if self.bnn and start_aleat:
                #End loss: Expected OP cost value based on pred distrib
                end_loss_ = self.end_loss_dist(y_preds, y_batch)
                kl_loss_ = self.K*self.model.kl_divergence_NN()/n_batches
                total_loss = end_loss_ + kl_loss_
                
            elif self.bnn and not start_aleat:
                y_preds = y_preds.mean(axis=0)
                end_loss_ = self.OP_simple.end_loss(y_preds, y_batch) 
                kl_loss_ = torch.tensor(0)
                total_loss = end_loss_ + kl_loss_                
                
            else:
                #End loss: OP cost value based on pred value
                end_loss_ = self.end_loss(y_preds, y_batch)                
                kl_loss_ = torch.tensor(0)
                total_loss = end_loss_ + kl_loss_
                
            total_loss.backward()
            self.opt.step()

            end_total_loss += end_loss_.item()
            kl_running_loss += kl_loss_.item()

        end_total_loss = end_total_loss/n_batches
        kl = kl_running_loss

        return end_total_loss, kl
    
    
    
    def train(self, EPOCHS=150):
        """
        Update ANN or BNN weights with Combined Learning approach 
        for EPOCHS epochs. 
        """
        epoch_number = 0
        
        best_loss = np.inf
        best_model = copy.deepcopy(self.model)
        
        es_count = 0
        es_count_trs = 40
        
        for epoch in tqdm(range(EPOCHS)):

            self.model.train(True)
            
            start_aleat = False
            if epoch > 50:
                start_aleat = True
            end_loss, kl_loss = self.train_one_epoch(start_aleat)
            total_loss = end_loss + kl_loss

            self.model.train(False)

            n = len(self.validation_loader.dataset)
            n_batches = len(self.validation_loader)
            
            total_running_loss_v = 0.0
            
            if self.bnn and start_aleat:
                kl_loss_val = self.K*self.model.kl_divergence_NN()
            else:
                kl_loss_val = torch.tensor(0)
            
            # Validation metrics
            for i, vdata in enumerate(self.validation_loader):

                x_val_batch, y_val_batch = vdata
                
                x_val_batch = x_val_batch.to(self.dev)
                y_val_batch = y_val_batch.to(self.dev)
                      
                y_val_preds, rho_val_preds = self.model(x_val_batch)

                if self.aleat_bool and start_aleat:                 
                    y_val_preds = y_val_preds + torch.sqrt(
                        torch.exp(rho_val_preds))*torch.randn(
                        y_val_preds.size(), device = self.dev)

                y_val_preds = self.inverse_transform(y_val_preds)
                y_val_batch = self.inverse_transform(y_val_batch)
                if self.bnn and start_aleat:
                    total_loss_v = self.end_loss_dist(y_val_preds, y_val_batch)

                elif self.bnn and not start_aleat: 
                    y_val_preds = y_val_preds.mean(axis=0)
                    total_loss_v = self.OP_simple.end_loss(y_val_preds, y_val_batch)
                    
                    
                else:
                    total_loss_v = self.end_loss(y_val_preds, y_val_batch)
                
                total_running_loss_v += total_loss_v.detach().item()
                                  
            
            end_loss_val = (total_running_loss_v/n_batches)
            kl_loss_val = (kl_loss_val).item()
            total_loss_val = end_loss_val + kl_loss_val
            
            if epoch_number == 0 or (epoch_number+1)%1 == 0: 
                print('------------------EPOCH {}------------------'.format(
                    epoch_number + 1))
                print(
                    f'END LOSS \t train {round(end_loss, 3)} valid {round(end_loss_val, 3)} \n',
                    f'KL LOSS \t train {round(kl_loss/(self.K+0.0001), 3)} valid {round(kl_loss_val/(self.K+0.0001), 3)} \n',
                    f'TOTAL LOSS \t train {round(total_loss, 3)} valid {round(total_loss_val, 3)} \n',
                )

            if  total_loss_val < best_loss:
                best_loss = total_loss_val
                best_model=copy.deepcopy(self.model)
  
            epoch_number += 1
    
            if epoch < 200:
                self.scheduler.step()
            
            if es_count>es_count_trs:
                print('EARLY STOP')
                break
            
        return best_model