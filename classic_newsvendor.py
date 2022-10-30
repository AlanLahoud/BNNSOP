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
from model import VariationalLayer, VariationalNet, StandardNet, WeakVariationalNet, WeakStandardNet
from train import TrainDecoupled, TrainCombined
from classical_newsvendor_utils import ClassicalNewsvendor


   
def run_classic_newsvendor(
            method_name, 
            method_learning,
            noise_type,
            seed_number,
            aleat_bool,
            N_SAMPLES,
            M_SAMPLES,
            dev):
    
    ##################################################################
    ##### Setting Parameters #########################################
    ##################################################################

    np.random.seed(seed_number)
    torch.manual_seed(seed_number)
    random.seed(seed_number)

    assert (method_name in ['ann','bnn'])
    assert (method_learning in ['decoupled','combined'])
    assert (noise_type in ['gaussian','multimodal'])
    assert (aleat_bool in [True, False])
    assert (N_SAMPLES>=1 and N_SAMPLES<9999)
    assert (M_SAMPLES>=1 and M_SAMPLES<9999)

    bnn = False 
    if method_name == 'bnn':
        bnn = True   
        K = 5 # Hyperparameter for the training in ELBO loss
        PLV = 3 # Hyperparameter for the prior in ELBO loss         

    model_name = method_name
    for i in range(2, len(sys.argv)):
        model_name += '_'+sys.argv[i]
    model_name += '_'+ str(seed_number)

    N_train = 1800
    N_valid = 1200
    N_test = 1200

    BATCH_SIZE_LOADER = 32 # Standard batch size
    EPOCHS = 200  # Epochs on training

    lr = 0.002
    
    #OP deterministic params
    cost_shortage=100
    if method_learning == 'combined':
        cost_excess=900 # quantile 0.1
        model_name = model_name \
        + f'_q{(cost_shortage)/(cost_shortage+cost_excess)}'
        cn = ClassicalNewsvendor(cost_shortage, cost_excess)
        lr = 0.0003


    ##################################################################
    ##### Data #######################################################
    ##################################################################

    X, y_original = data_generator.data_1to1(
        N_train, noise_level=1, noise_type = noise_type,
        uniform_input_space=False)

    # Output normalization
    scaler = StandardScaler()
    scaler.fit(y_original)
    tmean = torch.tensor(scaler.mean_.item())
    tstd = torch.tensor(scaler.scale_.item())
    joblib.dump(scaler, 'scaler.gz') # if you need to analyse

    def inverse_transform(yy):
        return yy*tstd + tmean

    y = scaler.transform(y_original).copy()
    #y = y_original
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    data_train = data_generator.ArtificialDataset(X, y)
    training_loader = torch.utils.data.DataLoader(
        data_train, batch_size=BATCH_SIZE_LOADER,
        shuffle=False, num_workers=mp.cpu_count())

    X_val, y_val_original = data_generator.data_1to1(
        N_valid, noise_level=1, noise_type = noise_type, 
        uniform_input_space=False)
    y_val = scaler.transform(y_val_original).copy()
    #y_val = y_val_original
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val_original = torch.tensor(y_val_original, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    data_valid = data_generator.ArtificialDataset(X_val, y_val)
    validation_loader = torch.utils.data.DataLoader(
        data_valid, batch_size=BATCH_SIZE_LOADER,
        shuffle=False, num_workers=mp.cpu_count())

    X_test, y_test_original = data_generator.data_1to1(
        N_test, noise_level=1, noise_type = noise_type, 
        uniform_input_space=False)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test_original = torch.tensor(y_test_original, dtype=torch.float32)
     
    input_size = X.shape[1]
    output_size = y.shape[1]



    ##################################################################
    ##### Model and Training #########################################
    ##################################################################
    
    #model_name = 'weak_' + model_name
    if method_name == 'bnn':
        #h = WeakVariationalNet(
        #    N_SAMPLES, input_size, output_size, PLV, dev).to(dev)
        h = VariationalNet(
            N_SAMPLES, input_size, output_size, PLV, dev).to(dev)

    elif method_name == 'ann':
        #h = WeakStandardNet(input_size, output_size).to(dev)
        h = StandardNet(input_size, output_size).to(dev)
        K = 0

    opt_h = torch.optim.Adam(h.parameters(), lr=0.0007)
    mse_loss = nn.MSELoss(reduction='none')

    if method_learning == 'decoupled':
        train_NN = TrainDecoupled(
                        bnn = bnn,
                        model=h,
                        opt=opt_h,
                        loss_data=mse_loss,
                        K=K,
                        aleat_bool=aleat_bool,
                        training_loader=training_loader,
                        validation_loader=validation_loader,
                        dev=dev
                    )

    elif method_learning == 'combined':
        train_NN = TrainCombined(
                        bnn = bnn,
                        model=h,
                        opt=opt_h,
                        aleat_bool=aleat_bool,
                        training_loader=training_loader,
                        scaler=scaler,
                        validation_loader=validation_loader,
                        OP=cn,
                        dev=dev
                    )

    else:
        print('check method_learning variable')
        quit()

    model_used = train_NN.train(EPOCHS=EPOCHS)


    M = M_SAMPLES
    model_used.update_n_samples(n_samples=M)
    y_pred = model_used.forward_dist(X_test, aleat_bool)[:,:,0]
    y_pred = inverse_transform(y_pred)


    mse_loss = nn.MSELoss()
    mse_loss_result = mse_loss(
        y_pred.mean(axis=0).squeeze(), 
        y_test_original.squeeze()
    ).item()

    ##################################################################
    ##### Solving the Optimization Problem ###########################
    ##################################################################

    dict_results = {}

    for quantile in np.arange(0.02,1,0.02):
        cost_excess = cost_shortage*(1-quantile)/quantile
        cn2 = ClassicalNewsvendor(cost_shortage, cost_excess)
        dict_results[str(round(quantile, 4))] = round(
            cn2.compute_norm_regret_from_preds(
            X_test, y_test_original, y_pred, M, method_name).item(), 
            3)
    
    print('Results for seed = ', seed_number)
    print('MSE loss: ', round(mse_loss_result, 5))
    print('REGRET: ', round(dict_results['0.1'], 5))#only for q=0.1

    return model_used, model_name, dict_results, mse_loss_result
    

if __name__ == '__main__':
    
    is_cuda = False
    dev = torch.device('cpu')  
    if torch.cuda.is_available():
        is_cuda = True
        dev = torch.device('cuda') 
    
    assert (len(sys.argv)==8)
    method_name = sys.argv[1] # ann or bnn
    method_learning = sys.argv[2] # decoupled or combined
    noise_type = sys.argv[3] # gaussian or multimodal
    nr_seeds = int(sys.argv[4]) # Average results through seeds
    aleat_bool = bool(int(sys.argv[5])) # Modeling aleatoric uncert
    N_SAMPLES = int(sys.argv[6])  # Sampling size while training
    M_SAMPLES = int(sys.argv[7])  # Sampling size while optimizing
    
    df_total = pd.DataFrame()
    for seed_number in range(0, nr_seeds):
        model_used, model_name, dict_results, mse_loss_result \
        = run_classic_newsvendor(
            method_name, 
            method_learning,
            noise_type,
            seed_number,
            aleat_bool,
            N_SAMPLES,
            M_SAMPLES,
            dev
        )
        
        df_results = pd.DataFrame.from_dict(
            dict_results, orient='index', columns=[f'REGRET_{seed_number}'])
        df_results[f'MSE_{seed_number}'] = mse_loss_result
        
        df_total = pd.concat([df_total, df_results], axis=1)
        
               
        ##############################################################
        ##### Saving model and results ###############################
        ##############################################################
        
        torch.save(model_used, 
                   f'./models/{model_name}_{seed_number}.pkl') 
         

    cols_mse = [c for c in df_total.columns.tolist() if 'MSE' in c]
    cols_nr = [c for c in df_total.columns.tolist() if 'REGRET' in c]
    
    print('---------------------------------------------------')
    print('-----------------Results---------------------------')
    print(f'Params: {model_name}')
    mse_mean = df_total[cols_mse].mean(axis=1).mean()
    mse_std = df_total[cols_mse].std(axis=1).mean()
    print('MSE: ', round(mse_mean, 5), '(', round(mse_std, 5), ')')
    
    nr01_mean = df_total[cols_nr].mean(axis=1).loc['0.1']
    nr01_std = df_total[cols_nr].std(axis=1).loc['0.1']
    print('REGRET 0.1: ', round(nr01_mean, 5), '(', round(nr01_std, 5), ')')
    

    df_total.to_csv(f'./newsvendor_results/{model_name}_nr.csv')