import numpy as np
import pandas as pd
import random
import joblib
import sys

import torch
import torch.multiprocessing as mp
import torch.nn as nn

from qpth.qp import QPFunction
from sklearn.preprocessing import StandardScaler

import data_generator
import params_newsvendor as params
from model import VariationalLayer, VariationalNet, StandardNet
from train import TrainDecoupled, TrainCombined
import constrained_newsvendor_utils as cnu

    
def run_constrained_newsvendor(
            method_name, 
            method_learning,
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
    assert (aleat_bool in [True, False])
    assert (N_SAMPLES>=1 and N_SAMPLES<9999)
    assert (M_SAMPLES>=1 and M_SAMPLES<9999)

    bnn = False 
    if method_name == 'bnn':
        bnn = True   
        K = 5 # Hyperparameter for the training in ELBO loss
        PLV = 3 # Hyperparameter for the prior in ELBO loss         

    model_name = method_name + '_constrained_'
    for i in range(2, len(sys.argv)):
        model_name += '_'+sys.argv[i]
    model_name += '_'+ str(seed_number)
        
    N_train = 4000
    N_valid = 2000
    N_test = 2000

    BATCH_SIZE_LOADER = 32 # Standard batch size
    EPOCHS = 150  # Epochs on training
    
    lr = 0.001
    
    if dev == torch.device('cuda'):
        BATCH_SIZE_LOADER = 128
    
    if method_learning == 'combined':
        EPOCHS = 40
        lr = 0.005



    ##################################################################
    ##### Data #######################################################
    ##################################################################

    nl=1
    X, Y_original = data_generator.data_4to8(
        N_train, noise_level=nl, 
        uniform_input_space=False)

    # Output normalization
    scaler = StandardScaler()
    scaler.fit(Y_original)
    tmean = torch.tensor(scaler.mean_).to(dev)
    tstd = torch.tensor(scaler.scale_).to(dev)
    joblib.dump(scaler, 'scaler_constrained.gz')

    def inverse_transform(yy):
        return yy*tstd + tmean

    #Y = scaler.transform(Y_original).copy()
    Y = Y_original
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)

    cpu_count = mp.cpu_count()
    if dev == torch.device('cuda'):
        print('Cuda found')
        cpu_count = 1
        
    data_train = data_generator.ArtificialDataset(X, Y)
    training_loader = torch.utils.data.DataLoader(
        data_train, batch_size=BATCH_SIZE_LOADER,
        shuffle=False, num_workers=cpu_count)

    X_val, Y_val_original = data_generator.data_4to8(
        N_valid, noise_level=nl, 
        uniform_input_space=False)
    #Y_val = scaler.transform(Y_val_original).copy()
    Y_val = Y_val_original
    X_val = torch.tensor(X_val, dtype=torch.float32)
    Y_val_original = torch.tensor(Y_val_original, dtype=torch.float32)
    Y_val = torch.tensor(Y_val, dtype=torch.float32)

    data_valid = data_generator.ArtificialDataset(X_val, Y_val)
    validation_loader = torch.utils.data.DataLoader(
        data_valid, batch_size=BATCH_SIZE_LOADER,
        shuffle=False, num_workers=cpu_count)

    X_test, Y_test_original = data_generator.data_4to8(
        N_test, noise_level=nl, 
        uniform_input_space=False)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(dev)
    Y_test_original = torch.tensor(Y_test_original, dtype=torch.float32).to(dev)

    input_size = X.shape[1]
    output_size = Y.shape[1]

    #OP deterministic params
    n_items = output_size
    params_t, _ = params.get_params(n_items, seed_number, dev)

    # Construct the solver
    op_solver = cnu.SolveConstrainedNewsvendor(params_t, 1, dev)
    op_solver_dist = cnu.SolveConstrainedNewsvendor(params_t, N_SAMPLES, dev)

    if not aleat_bool and method_name=='ann':
        op_solver_dist = op_solver

    ##################################################################
    ##### Model and Training #########################################
    ##################################################################

    if method_name == 'bnn':
        h = VariationalNet(
            N_SAMPLES, input_size, output_size, PLV, dev).to(dev)

    elif method_name == 'ann':
        h = StandardNet(input_size, output_size).to(dev)
        K = 0

    opt_h = torch.optim.Adam(h.parameters(), lr=lr)
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
                        validation_loader=validation_loader,
                        OP=op_solver_dist,
                        dev=dev
                    )

    else:
        print('check method_learning variable')
        quit()


    model_used = train_NN.train(EPOCHS=EPOCHS)

    model_used.update_n_samples(n_samples=M_SAMPLES)
    Y_pred = model_used.forward_dist(X_test, aleat_bool)
    #Y_pred = inverse_transform(Y_pred)

    mse_loss = nn.MSELoss()
    mse_loss_result = mse_loss(Y_pred.mean(axis=0), Y_test_original).item()




    ##################################################################
    ##### Solving the Optimization Problem ###########################
    ##################################################################
    
    # Construct the solver again for the optimization part
    op_solver = cnu.SolveConstrainedNewsvendor(params_t, 1, dev)
    op_solver_dist = cnu.SolveConstrainedNewsvendor(params_t, M_SAMPLES, dev)
    if not aleat_bool and method_name=='ann':
        op_solver_dist = op_solver

    n_batches = int(np.ceil(
        Y_pred.shape[1]/BATCH_SIZE_LOADER))
    f_total = 0
    for b in range(0, n_batches):
        i_low = b*BATCH_SIZE_LOADER
        i_up = (b+1)*BATCH_SIZE_LOADER
        if b == n_batches-1:
            i_up = n_batches*Y_pred.shape[1]
        f_total += op_solver_dist.cost_fn(
            Y_pred[:,i_low:i_up,:], 
            Y_test_original[i_low:i_up,:])/n_batches


    n_batches = int(np.ceil(
        Y_test_original.shape[0]/BATCH_SIZE_LOADER))
    f_total_best = 0
    for b in range(0, n_batches):
        i_low = b*BATCH_SIZE_LOADER
        i_up = (b+1)*BATCH_SIZE_LOADER
        if b == n_batches-1:
            i_up = n_batches*Y_test_original.shape[0]
        f_total_best += op_solver.cost_fn(
            Y_test_original[i_low:i_up,:].unsqueeze(0), 
            Y_test_original[i_low:i_up,:])/n_batches

    nr_result = (f_total.item() - f_total_best.item())/f_total_best.item()

    print('Results for seed = ', seed_number)
    print('MSE loss: ', round(mse_loss_result, 5))
    print('END cost: ', round(f_total.item(), 5))
    print('BEST cost: ', round(f_total_best.item(), 5))
    print('NR: ', round(nr_result, 5))

    return model_used, model_name, nr_result, mse_loss_result

if __name__ == '__main__':
    
    is_cuda = False
    dev = torch.device('cpu')  
    if torch.cuda.is_available():
        is_cuda = True
        dev = torch.device('cuda') 

        
    assert (len(sys.argv)==7)
    method_name = sys.argv[1] # ann or bnn
    method_learning = sys.argv[2] # decoupled or combined
    nr_seeds = int(sys.argv[3]) # Average results through seeds
    aleat_bool = bool(int(sys.argv[4])) # Modeling aleatoric uncert
    N_SAMPLES = int(sys.argv[5])  # Sampling size while training
    M_SAMPLES = int(sys.argv[6])  # Sampling size while optimizing
        
    mse_results = []
    nr_results = []
    for seed_number in range(0, nr_seeds):
        model_used, model_name, nr_result, mse_loss_result \
        = run_constrained_newsvendor(
            method_name, 
            method_learning,
            seed_number,
            aleat_bool,
            N_SAMPLES,
            M_SAMPLES,
            dev
        )
        
        mse_results.append(mse_loss_result)
        nr_results.append(nr_result)
        
        ##########################################################
        ##### Saving model and results ###########################
        ##########################################################
    
        torch.save(model_used, f'./models/{model_name}.pkl')  
    
    df_total = pd.DataFrame(data={
        'MSE':mse_results, 'NR':nr_results})
    
    mse_avg = df_total['MSE'].mean()
    mse_std = df_total['MSE'].std()
    
    nr_avg = df_total['NR'].mean()
    nr_std = df_total['NR'].std()
    
    print('---------------------------------------------------')
    print('-----------------Results---------------------------')
    print(f'Params: {model_name}')
    print('MSE: ', round(mse_avg, 5), '(', round(mse_std, 5), ')')
    print('NR: ', round(nr_avg, 5), '(', round(nr_std, 5), ')')
        
    df_total.to_csv(f'./newsvendor_results/{model_name}_nr.csv')