import sys

import numpy as np
import math
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torch.multiprocessing as mp
from datetime import datetime

from sklearn.model_selection import train_test_split

# Utils
import data_generator
from model import VariationalLayer, VariationalNet

from train import TrainDecoupledElbo



is_cuda = False
dev = torch.device('cpu')  
if torch.cuda.is_available():
    is_cuda = True
    dev = torch.device('cuda')  

    
# Setting the seeds to allow replication
# Changing the seed might require hyperparameter tuning again
# Because it changes the deterministic parameters
seed_number = 0
np.random.seed(seed_number)
torch.manual_seed(seed_number)
random.seed(seed_number)

if len(sys.argv)!=3:
    print('Please provide both K and PLV arguments')
    quit()
    
# Mandatory to set parameters
K = float(sys.argv[1]) # Regularization parameter (ELBO)
PLV = float(sys.argv[2]) # Variance of prior gaussian 

if K>50 or K<0:
    print('Try K between 0 and 50')
    quit()

if PLV<-3 or PLV>6:
    print('Try PLV between -3 and 6')
    quit()

# Setting parameters (change if necessary)
N = 10000 # Total data size
N_train = 6000 # Training data size
N_SAMPLES = 16 # Sampling size while training
BATCH_SIZE_LOADER = 32 # Standard batch size
EPOCHS = 100 

# Data manipulation
N_valid = N - N_train
X, y = data_generator.data_1to1(N)
X, y_perfect = data_generator.data_1to1(N, noise_level=1)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)
dataset = data_generator.ArtificialDataset(X, y)

data_train, data_valid = torch.utils.data.random_split(
    dataset, [N_train, N_valid], 
    generator=torch.manual_seed(seed_number))

training_loader = torch.utils.data.DataLoader(
    data_train, batch_size=BATCH_SIZE_LOADER,
    shuffle=False, num_workers=mp.cpu_count())

validation_loader = torch.utils.data.DataLoader(
    data_valid, batch_size=BATCH_SIZE_LOADER,
    shuffle=False, num_workers=mp.cpu_count())

input_size = X.shape[1]
output_size = y.shape[1]


# Model setting
h = VariationalNet(N_SAMPLES, input_size, output_size, PLV).to(dev)
opt_h = torch.optim.Adam(h.parameters(), lr=0.005)
mse_loss_mean = nn.MSELoss(reduction='mean')

# Training regression with ELBO
train_elbo = TrainDecoupledElbo(
                model=h,
                opt=opt_h,
                loss_data=mse_loss_mean,
                K=K,
                training_loader=training_loader,
                validation_loader=validation_loader
            )

train_elbo.train(EPOCHS=EPOCHS)

# Saving model
Kstr = str(K).replace('.','')
plvstr = str(PLV).replace('.','')
torch.save(train_elbo.model, f'./models/elbo_nv1_{Kstr}_{plvstr}.pkl')