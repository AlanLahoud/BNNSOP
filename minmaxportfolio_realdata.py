import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from gauss_proc import GP

import joblib
import random
import sys

import data_generator
#from model import VariableStandardNet, VariableVariationalNet
from model import POStandardNet, POVariationalNet
from train import TrainDecoupled, TrainCombined

from sklearn.preprocessing import StandardScaler

import torch.multiprocessing as mp

import minmax_op_utils as op_utils

from tqdm import tqdm




def run_minimax_op(
            method_name, 
            method_learning,
            seed_number,
            N_SAMPLES,
            M_SAMPLES,
            N_train,
            EPOCHS,
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
    
    K = .001
    PLV = 1

    bnn = False 
    if method_name == 'bnn':
        bnn = True   


    model_name = method_name + '_constrained_'
    for i in range(2, len(sys.argv)):
        model_name += '_'+sys.argv[i]
    model_name += '_'+ str(seed_number)
       
    
    BATCH_SIZE_LOADER = 128
    
    warm_decoupled = False
      
    if method_learning == 'decoupled' and method_name == 'ann':
        lr = 0.00005
        EPOCHS = 100
        pt = -1
    if method_learning == 'decoupled' and method_name == 'bnn':
        lr = 0.00001
        EPOCHS = EPOCHS
        pt = -1
    if method_learning == 'combined' and method_name == 'ann':
        lr = 0.001
        EPOCHS = 100
        pt = -1
    if method_learning == 'combined' and method_name == 'bnn':
        warm_decoupled = False
        lr = 0.0001
        EPOCHS = EPOCHS
        pt = -1
      
    # Aleatoric Uncertainty Modeling
    aleat_bool=True
    if method_name == 'ann':
        aleat_bool=False
        
    cpu_count = mp.cpu_count()
    if dev == torch.device('cuda'):
        print('Cuda found')
        cpu_count = 1

        
    N_train = N_train
    N_val = 3*N_train//5
    N_test = 1500

    n_samples_orig = 500

    # Noise level in the portfolio (aleatoric uncertainty)
    nl = 0.05

    # Generating assets dataset
    X_total, Y_total, _ = data_generator.gen_processed_stocks()
    
    X, Y_original = X_total[:800], Y_total[:800]
    X_val, Y_val_original = X_total[800:1200], Y_total[800:1200]
    X_test, Y_test_original = X_total[1200:], Y_total[1200:]
      
    # Output normalization
    scaler = StandardScaler()
    scaler.fit(Y_original)
    tmean = torch.tensor(scaler.mean_).to(dev)
    tstd = torch.tensor(scaler.scale_).to(dev)
    joblib.dump(scaler, 'scaler_portfolio.gz')
    
    # Input normalization
    scaler_X = StandardScaler()
    scaler_X.fit(X)

    # Function to denormalize the data
    def inverse_transform(yy):
        return yy*tstd + tmean

    Y = scaler.transform(Y_original).copy()
    X = scaler_X.transform(X).copy()
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    data_train = data_generator.ArtificialDataset(X, Y)
    training_loader = torch.utils.data.DataLoader(
        data_train, batch_size=BATCH_SIZE_LOADER,
        shuffle=True, num_workers=cpu_count)
    #Y_dist = torch.tensor(Y_dist, dtype=torch.float32)

    Y_val = scaler.transform(Y_val_original).copy()
    X_val = scaler_X.transform(X_val).copy()
    X_val = torch.tensor(X_val, dtype=torch.float32)
    Y_val_original = torch.tensor(Y_val_original, dtype=torch.float32)
    Y_val = torch.tensor(Y_val, dtype=torch.float32)
    data_valid = data_generator.ArtificialDataset(X_val, Y_val)
    validation_loader = torch.utils.data.DataLoader(
        data_valid, batch_size=BATCH_SIZE_LOADER,
        shuffle=False, num_workers=cpu_count)

    X_test = scaler_X.transform(X_test).copy()
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
    min_return = 1
    
    mse_loss = nn.MSELoss(reduction='none')
    op = op_utils.RiskPortOP(N_SAMPLES, N_ASSETS, min_return, torch.tensor(Y_original) + 0.1, dev)
    
    #GP Baseline model
    if method_name == 'gp':
        assert N_ASSETS >=1
        model_gps = []
        for k in tqdm(range(0, N_ASSETS)):
            gp = GP(length_scale=1, length_scale_bounds=(1e-2, 1e4), 
                    alpha_noise=0.01, white_noise=1, 
                    n_restarts_optimizer=12)
            gp.gp_fit(X.detach().numpy(), Y[:,k].detach().numpy())
            model_gps.append(gp)
            model_used = gp
    
    else:
        if method_name == 'bnn':
            h = POVariationalNet(
            n_samples=N_SAMPLES,
            input_size=X.shape[1], 
            output_size=Y.shape[1], 
            plv=PLV, 
            dev=dev
            ).to(dev)


        #ANN Baseline model
        elif method_name == 'ann':
            h = POStandardNet(X.shape[1], Y.shape[1]).to(dev)
            K = 0 # There is no K in ANN
            N_SAMPLES = 1

        opt_h = torch.optim.Adam(h.parameters(), lr=lr)  
        
        if method_name == 'ann':
            opt_h = torch.optim.Adam(h.parameters(), lr=lr, weight_decay=10e-3)        

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
                            dev=dev,
                            bm_stop=False
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
    
    op_true = op_utils.RiskPortOP(1, N_ASSETS, min_return, torch.tensor(Y_original), dev)
    
    opt_cost = 0
    for i, data in enumerate(test_loader):
        _ , y_batch = data
        opt_cost_ = op_true.end_loss_dist(y_batch.unsqueeze(0).to(dev), y_batch.to(dev), True).detach()
        opt_cost += opt_cost_
    
    opt_cost = opt_cost/len(test_loader)
    
    for M_opt in M_SAMPLES:
               
        if method_name == 'gp':
            for model in model_gps:
                model.update_n_samples(n_samples=M_opt)
        else:
            model_used.update_n_samples(n_samples=M_opt)
        
        op = op_utils.RiskPortOP(M_opt, N_ASSETS, min_return, torch.tensor(Y_original), dev)
                
        final_cost = 0
        for i, data in enumerate(test_loader):
            x_batch, y_batch = data
            x_batch = x_batch.to(dev)
            y_batch = y_batch.to(dev)
            
            if method_name in ['ann','bnn']:
                Y_pred = model_used.forward_dist(x_batch, aleat_bool)
                
            elif method_name in ['gp']:
                Y_pred = torch.zeros_like(
                    y_batch).unsqueeze(0).expand(
                    M_opt, y_batch.shape[0], 
                    y_batch.shape[1]).clone()
                for k in range(0, len(model_gps)):
                    Y_pred[:,:,k] = model_gps[k].forward_dist(
                        x_batch, aleat_bool).squeeze()
                
            else:
                print('Model not found')
                break
    
            
            Y_pred_original_ = inverse_transform(Y_pred)
     
            if method_learning == 'decoupled':
                final_cost_ = op.end_loss_dist(Y_pred_original_.to(dev), y_batch.to(dev), True).detach()               
            else:
                final_cost_ = op.end_loss_dist(Y_pred_original_.to(dev), y_batch.to(dev)).detach()
            final_cost += final_cost_
            
        final_cost = final_cost/len(test_loader)
           
        fc_list.append(final_cost.item())
        oc_list.append(opt_cost.item())
        
        print(f'Final cost: {final_cost} \t Opt cost: {opt_cost}')
        
    return fc_list, oc_list
    

if __name__ == '__main__':
    
    is_cuda = False
    dev = torch.device('cpu')  
    if torch.cuda.is_available():
        is_cuda = True
        dev = torch.device('cuda') 

        
    assert (len(sys.argv)==5 or len(sys.argv)==6 or len(sys.argv)==7)
    method_name = sys.argv[1] # ann or bnn or gp
    method_learning = sys.argv[2] # decoupled or combined
    nr_seeds = int(sys.argv[3]) # Average results through nr seeds
    N_SAMPLES = int(sys.argv[4])  # Sampling size while training (M_train)
    M_SAMPLES = [64, 32, 16, 8, 4, 2, 1] # Sampling size while optimizing (M_opt)
    #M_SAMPLES = [8, 4, 2, 1] # Sampling size while optimizing (M_opt)
    
    N_train = 800
    try:
        N_train = int(sys.argv[6])
    except:
        print(f'No training size provided, training with {N_train} samples')
        
        
    EPOCHS = 150
    try:
        EPOCHS = int(sys.argv[7])
    except:
        print(f'No EPOCH size provided, training with {EPOCHS} EPOCHS for BNNs')
    
    fc_l = []
    sc_l = []
    oc_l = []
    for seed_number in range(0, nr_seeds):
    
        fc_list, oc_list = run_minimax_op(
                                        method_name, 
                                        method_learning,
                                        seed_number,
                                        N_SAMPLES,
                                        M_SAMPLES,
                                        N_train,
                                        EPOCHS,
                                        dev)
        
        fc_l.append(fc_list)
        oc_l.append(oc_list)
            
    print('Costs', np.array(fc_l))        
    print('Rs', np.array(fc_l) - np.array(oc_l))
    
    print('Rmean',  (np.array(fc_l) - np.array(oc_l)).mean(0) )
    
    print('Rstd',  (np.array(fc_l) - np.array(oc_l)).std(0) )
    
    