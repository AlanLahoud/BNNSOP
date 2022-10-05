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
from sklearn.preprocessing import StandardScaler

import joblib

# Utils
import data_generator
from model import VariationalLayer, VariationalNet, StandardNet, VariationalNet2
from train import TrainDecoupled
from classical_newsvendor_utils import ClassicalNewsvendor
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

bnn = False
if method_name == 'bnn':
    assert (len(sys.argv)==3)
    bnn = True
    K = float(sys.argv[2])
    PLV = 5
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
for i in range(2, len(sys.argv)):
    model_name += sys.argv[i]


# Setting parameters (change if necessary)
N = 8000 # Total data size
N_train = 5000 # Training data size
N_SAMPLES = 16 # Sampling size while training
BATCH_SIZE_LOADER = 32 # Standard batch size
EPOCHS = 10 
noise_type = 'poisson'

# Data manipulation
N_valid = N - N_train
X, y_original = data_generator.data_1to1(N_train, noise_level=1, noise_type = noise_type)

# Output normalization
scaler = StandardScaler()
scaler.fit(y_original)
tmean = torch.tensor(scaler.mean_.item())
tstd = torch.tensor(scaler.scale_.item())
joblib.dump(scaler, 'scaler.gz')

def inverse_transform(yy):
    return yy*tstd + tmean

y = scaler.transform(y_original).copy()
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

data_train = data_generator.ArtificialDataset(X, y)
training_loader = torch.utils.data.DataLoader(
    data_train, batch_size=BATCH_SIZE_LOADER,
    shuffle=False, num_workers=mp.cpu_count())

X_val, y_val_original = data_generator.data_1to1(N_valid, noise_level=1, noise_type = noise_type)
y_val = scaler.transform(y_val_original).copy()
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val_original = torch.tensor(y_val_original, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

data_valid = data_generator.ArtificialDataset(X_val, y_val)
validation_loader = torch.utils.data.DataLoader(
    data_valid, batch_size=BATCH_SIZE_LOADER,
    shuffle=False, num_workers=mp.cpu_count())

input_size = X.shape[1]
output_size = y.shape[1]

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
    train_NN.model.update_n_samples(n_samples=M)
    y_pred = train_NN.model.forward_dist(X_val)[:,:,0]
    y_pred = inverse_transform(y_pred)
    
    for cost_price in (np.arange(0.1,1,0.1)*sell_price):
        quantile = (sell_price-cost_price)/sell_price
        cn2 = ClassicalNewsvendor(sell_price, cost_price)
        dict_results_nr[str(quantile)] = round(
            cn2.compute_norm_regret_from_preds(
            X_val, y_val_original, y_pred, M, method_name).item(), 
            3)


else:
    # Training regression with FLOW
    trfl = train_normflow.TrainFlowDecoupled(steps = 5000, input_size=1, output_size=1)
    pyx = trfl.train(X, y, X_val, y_val)
    model_used = pyx
    
    # Propagating predictions to Newsvendor Problem
    M = 1000
    N = X_val.shape[0]     
    y_pred = torch.zeros((M, N))
    for i in range(0, N):
        y_pred[:,i] = pyx.condition(X_val[i]).sample(torch.Size([M,])).squeeze()
    
    y_pred = inverse_transform(y_pred)
    sell_price = 200
    dict_results_nr = {}
    for cost_price in (np.arange(0.1,1,0.05)*sell_price):
        quantile = (sell_price-cost_price)/sell_price
        cn2 = ClassicalNewsvendor(sell_price, cost_price)
        dict_results_nr[str(quantile)] = round(
            cn2.compute_norm_regret_from_preds(
            X_val, y_val_original, y_pred, M, method_name).item(), 
            3)             

torch.save(model_used, f'./models/{model_name}_{noise_type}.pkl')
    
    
df_nr = pd.DataFrame.from_dict(dict_results_nr, orient='index', columns=['NR'])
df_nr.to_csv(f'./newsvendor_results/{model_name}_{noise_type}_nr.csv')