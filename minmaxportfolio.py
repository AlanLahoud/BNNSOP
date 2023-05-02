import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import joblib
import random
import sys

import data_generator
#from model import VariableStandardNet, VariableVariationalNet
from model import StrongStandardNet, StrongVariationalNet
from train import TrainDecoupled, TrainCombined

from sklearn.preprocessing import StandardScaler

import torch.multiprocessing as mp

import minmax_op_utils as op_utils




def run_minimax_op(
            method_name, 
            method_learning,
            seed_number,
            N_SAMPLES,
            M_SAMPLES,
            N_ASSETS,
            dev):


    ##################################################################
    ##### Setting Parameters #########################################
    ##################################################################

    np.random.seed(seed_number)
    torch.manual_seed(seed_number)
    random.seed(seed_number)

    assert (method_name in ['ann','bnn','gp'])
    assert (method_learning in ['decoupled','combined'])
    assert (N_SAMPLES>=1 and N_SAMPLES<9999)
    
    K = 1
    PLV = 1

    bnn = False 
    if method_name == 'bnn':
        bnn = True   


    model_name = method_name + '_constrained_'
    for i in range(2, len(sys.argv)):
        model_name += '_'+sys.argv[i]
    model_name += '_'+ str(seed_number)
       
    
    EPOCHS = 200  # Epochs on training   
    BATCH_SIZE_LOADER = 64
    
    warm_decoupled = False
      
    if method_learning == 'decoupled' and method_name == 'ann':
        lr = 0.001
        EPOCHS = 100
        pt = -1
    if method_learning == 'decoupled' and method_name == 'bnn':
        lr = 0.0005
        EPOCHS = 200
        pt = -1
    if method_learning == 'combined' and method_name == 'ann':
        lr = 0.0005
        EPOCHS = 200
        pt = -1
    if method_learning == 'combined' and method_name == 'bnn':
        warm_decoupled = False
        lr = 0.0005
        EPOCHS = 200
        pt = -1
      
    # Aleatoric Uncertainty Modeling
    aleat_bool=True
    if method_name == 'ann':
        aleat_bool=False
        
    cpu_count = mp.cpu_count()
    if dev == torch.device('cuda'):
        print('Cuda found')
        cpu_count = 1

        
    N_train = 2500
    N_val = 1500
    N_test = 2500

    n_samples_orig = 100

    # Noise level in the portfolio (aleatoric uncertainty)
    nl = 0.05

    # Generating assets dataset
    X, Y_original, _ = data_generator.gen_data(N_train, N_ASSETS, nl, seed_number, n_samples_orig)
    X_val, Y_val_original, _ = data_generator.gen_data(N_val, N_ASSETS, nl, seed_number + 80, n_samples_orig)
    X_test, Y_test_original, Y_test_dist = data_generator.gen_data(N_test, N_ASSETS, nl, seed_number + 160, n_samples_orig)
      
    # Output normalization
    scaler = StandardScaler()
    scaler.fit(Y_original)
    tmean = torch.tensor(scaler.mean_).to(dev)
    tstd = torch.tensor(scaler.scale_).to(dev)
    joblib.dump(scaler, 'scaler_portfolio.gz')

    # Function to denormalize the data
    def inverse_transform(yy):
        return yy*tstd + tmean

    Y = scaler.transform(Y_original).copy()
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    data_train = data_generator.ArtificialDataset(X, Y)
    training_loader = torch.utils.data.DataLoader(
        data_train, batch_size=BATCH_SIZE_LOADER,
        shuffle=True, num_workers=cpu_count)
    #Y_dist = torch.tensor(Y_dist, dtype=torch.float32)

    Y_val = scaler.transform(Y_val_original).copy()
    X_val = torch.tensor(X_val, dtype=torch.float32)
    Y_val_original = torch.tensor(Y_val_original, dtype=torch.float32)
    Y_val = torch.tensor(Y_val, dtype=torch.float32)
    data_valid = data_generator.ArtificialDataset(X_val, Y_val)
    validation_loader = torch.utils.data.DataLoader(
        data_valid, batch_size=BATCH_SIZE_LOADER,
        shuffle=False, num_workers=cpu_count)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_test_original = torch.tensor(
        Y_test_original, dtype=torch.float32)
    data_test = data_generator.ArtificialDataset(
        X_test, Y_test_original)
    test_loader = torch.utils.data.DataLoader(
    data_test, batch_size=BATCH_SIZE_LOADER,
    shuffle=False, num_workers=cpu_count) 
    
    
    n_samples = Y.shape[0]
    N_ASSETS = Y.shape[1]
    min_return = 100
    
    
    if method_name == 'bnn':
        h = StrongVariationalNet(
        n_samples=N_SAMPLES,
        input_size=X.shape[1], 
        output_size=Y.shape[1], 
        plv=PLV, 
        dev=dev
        ).to(dev)

    
    #ANN Baseline model
    elif method_name == 'ann':
        h = StrongStandardNet(X.shape[1], Y.shape[1]).to(dev)
        K = 0 # There is no K in ANN
        N_SAMPLES = 1

    opt_h = torch.optim.Adam(h.parameters(), lr=lr)
    mse_loss = nn.MSELoss(reduction='none')

    op = op_utils.RiskPortOP(N_SAMPLES, N_ASSETS, min_return, torch.tensor(Y_original), dev)

    # Decoupled learning approach
    if method_learning == 'decoupled' or warm_decoupled:
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

    # Combined learning approach (end-to-end loss)
    elif method_learning == 'combined':
        train_NN = TrainCombined(
                        bnn = bnn,
                        model=h,
                        opt=opt_h,
                        K=K,
                        aleat_bool=aleat_bool,
                        training_loader=training_loader,
                        scaler=scaler,
                        validation_loader=validation_loader,
                        OP=op,
                        dev=dev
                    )
        

    if warm_decoupled:
        EPOCHS1 = 30
    else:
        EPOCHS1 = EPOCHS
    model_used = train_NN.train(EPOCHS=EPOCHS1, pre_train=pt)
    
    if warm_decoupled:
        train_NN = TrainCombined(
                        bnn = bnn,
                        model=model_used,
                        opt=opt_h,
                        K=K,
                        aleat_bool=aleat_bool,
                        training_loader=training_loader,
                        scaler=scaler,
                        validation_loader=validation_loader,
                        OP=op,
                        dev=dev
                    )
    
        model_used = train_NN.train(EPOCHS=EPOCHS-EPOCHS1, pre_train=pt)
    
    
    if method_name == 'ann':
        M_SAMPLES = [1]
    
    fc_list = []
    sc_list = []
    oc_list = []
    
    op = op_utils.RiskPortOP(n_samples_orig, N_ASSETS, min_return, torch.tensor(Y_original), dev)
    subopt_cost = op.end_loss_dist(torch.tensor(Y_test_dist).to(dev), Y_test_original.to(dev))
    
    op = op_utils.RiskPortOP(1, N_ASSETS, min_return, torch.tensor(Y_original), dev)
    opt_cost = op.end_loss_dist(Y_test_original.unsqueeze(0).to(dev), Y_test_original.to(dev))
    
    for M_opt in M_SAMPLES:
        model_used.update_n_samples(n_samples=M_opt)
        Y_pred = model_used.forward_dist(X_test, aleat_bool)
        Y_pred_original = inverse_transform(Y_pred)
        
        op = op_utils.RiskPortOP(M_opt, N_ASSETS, min_return, torch.tensor(Y_original), dev)
        final_cost = op.end_loss_dist(Y_pred_original.to(dev), Y_test_original.to(dev))
           
        fc_list.append(final_cost.item())
        sc_list.append(subopt_cost.item())
        oc_list.append(opt_cost.item())
        
        print(f'Final cost: {final_cost} \t Subopt cost: {subopt_cost} \t Opt cost: {opt_cost}')
        
    return fc_list, sc_list, oc_list
    

if __name__ == '__main__':
    
    is_cuda = False
    dev = torch.device('cpu')  
    if torch.cuda.is_available():
        is_cuda = True
        dev = torch.device('cuda') 

        
    assert (len(sys.argv)==6)
    method_name = sys.argv[1] # ann or bnn or gp
    method_learning = sys.argv[2] # decoupled or combined
    nr_seeds = int(sys.argv[3]) # Average results through nr seeds
    N_SAMPLES = int(sys.argv[4])  # Sampling size while training (M_train)
    M_SAMPLES = [64, 32, 16, 8, 4, 2, 1] # Sampling size while optimizing (M_opt)
    N_ASSETS = int(sys.argv[5])  # Sampling size while training (M_train)
    #M_SAMPLES = [8, 4, 2, 1] # Sampling size while optimizing (M_opt)
    
    fc_l = []
    sc_l = []
    oc_l = []
    for seed_number in range(0, nr_seeds):
    
        fc_list, sc_list, oc_list = run_minimax_op(
                                        method_name, 
                                        method_learning,
                                        seed_number,
                                        N_SAMPLES,
                                        M_SAMPLES,
                                        N_ASSETS,
                                        dev)
        
        fc_l.append(fc_list)
        sc_l.append(sc_list)
        oc_l.append(oc_list)
            
    print('Costs', np.array(fc_l))        
    print('FRs', np.array(fc_l) - np.array(sc_l))
    print('Rs', np.array(fc_l) - np.array(oc_l))
    
    print('FRmean',  (np.array(fc_l) - np.array(sc_l)).mean(0) )
    print('Rmean',  (np.array(fc_l) - np.array(oc_l)).mean(0) )
    
    print('FRstd',  (np.array(fc_l) - np.array(sc_l)).std(0) )
    print('Rstd',  (np.array(fc_l) - np.array(oc_l)).std(0) )
    
    