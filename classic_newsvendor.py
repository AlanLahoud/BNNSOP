import sys

import pandas as pd
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
from model import VariationalLayer, VariationalNet, StandardNet, VariationalNet2
from train import TrainDecoupled
import classical_newsvendor_utils as cnu
import train_normflow

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

method_name = sys.argv[1]

if method_name == 'bnn':
    assert (len(sys.argv)==3)
    K = float(sys.argv[2])
    PLV = 1
    if K>10000 or K<0:
        print('Try K between 0 and 10000')
        quit()    
elif method_name == 'flow':
    assert (len(sys.argv)==2)
elif method_name == 'ann':
    assert (len(sys.argv)==3)
    eps = float(sys.argv[2])
else:
    print('Method not implemented: Try bnn or flow')
    quit()
    
model_name = method_name
for i in range(1, len(sys.argv)-1):
    model_name += sys.argv[i]


# Setting parameters (change if necessary)
N = 8000 # Total data size
N_train = 5000 # Training data size
N_SAMPLES = 8 # Sampling size while training
BATCH_SIZE_LOADER = 64 # Standard batch size
EPOCHS = 350 
noise_type = 'multimodal'

# Data manipulation
N_valid = N - N_train
X, y = data_generator.data_1to1(N, noise_level=1, noise_type = noise_type)
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

X_val = validation_loader.dataset.dataset.X[validation_loader.dataset.indices]
y_val = validation_loader.dataset.dataset.y[validation_loader.dataset.indices]

if method_name != 'flow':

    # Model setting
    if method_name == 'bnn':
        h = VariationalNet2(N_SAMPLES, input_size, output_size, PLV).to(dev)
        
    elif method_name == 'ann':
        h = StandardNet(input_size, output_size, eps).to(dev)
        K = 0

    opt_h = torch.optim.Adam(h.parameters(), lr=0.001)
    mse_loss = nn.MSELoss(reduction='none')

    # Training regression with BNN or ANN
    train_NN = TrainDecoupled(
                    bnn = bnn,
                    model=h,
                    opt=opt_h,
                    loss_data=mse_loss,
                    K=K,
                    training_loader=training_loader,
                    validation_loader=validation_loader
                )

    train_NN.train(EPOCHS=EPOCHS)
    model_used = train_NN.model
    
    # Propagating predictions to Newsvendor Problem
    M = 1000
    sell_price = 200
    dict_results_nr = {}
    for cost_price in (np.arange(0.1,1,0.1)*sell_price):
        quantile = (sell_price-cost_price)/sell_price
        dict_results_nr[str(quantile)] = round(
            cnu.compute_norm_regret(
            X_val, y_val, train_elbo.model, M, sell_price, cost_price).item(), 
            3)


else:
    # Training regression with FLOW
    trfl = train_normflow.TrainFlowDecoupled(steps = 5000)
    pyx = trfl.train(X, y, X_val, y_val)
    model_used = pyx
    
    # Propagating predictions to Newsvendor Problem
    M = 1000
    N = X_val.shape[0]     
    y_pred = torch.zeros((M, N))
    for i in range(0, N):
        y_pred[:,i] = pyx.condition(X_val[i]).sample(torch.Size([M,])).squeeze()
        
    sell_price = 200
    dict_results_nr = {}
    for cost_price in (np.arange(0.1,1,0.1)*sell_price):
        quantile = (sell_price-cost_price)/sell_price
        dict_results_nr[str(quantile)] = round(
            cnu.compute_norm_regret_from_preds(
            X_val, y_val, y_pred, M, sell_price, cost_price).item(), 
            3)             

torch.save(model_used, f'./models/{model_name}_{noise_type}.pkl')
    
    
df_nr = pd.DataFrame.from_dict(dict_results_nr, orient='index', columns=['NR'])
df_nr.to_csv(f'./newsvendor_results/{model_name}_{noise_type}_nr.csv')