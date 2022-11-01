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
from model import VariationalLayer, VariationalNet, StandardNet, StrongVariationalNet, WeakVariationalNet, WeakStandardNet
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
    #assert (M_SAMPLES>=1 and M_SAMPLES<9999)

    bnn = False 
    if method_name == 'bnn':
        bnn = True   
        K = 1 # Hyperparameter for the training in ELBO loss
        PLV = 1 # Hyperparameter for the prior in ELBO loss         

    model_name = method_name + '_constrained_'
    for i in range(2, len(sys.argv)):
        model_name += '_'+sys.argv[i]
    model_name += '_'+ str(seed_number)
        
    N_train = 6000
    N_valid = 3000
    N_test = 3000

    BATCH_SIZE_LOADER = 32 # Standard batch size
    EPOCHS = 150  # Epochs on training
    
    
    if dev == torch.device('cuda'):
        BATCH_SIZE_LOADER = 256
    
    lr = 0.001
    if method_learning == 'decoupled' and method_name == 'ann':
        lr = 0.001
    if method_learning == 'decoupled' and method_name == 'bnn':
        lr = 0.0005
    if method_learning == 'combined' and method_name == 'ann':
        lr = 0.0005
    if method_learning == 'combined' and method_name == 'bnn':
        lr = 0.0001



    ##################################################################
    ##### Data #######################################################
    ##################################################################

    nl=0.5
    X, Y_original, _ = data_generator.data_4to8(
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

    Y = scaler.transform(Y_original).copy()
    #Y = Y_original
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

    
    X_val, Y_val_original, _ = data_generator.data_4to8(
        N_valid, noise_level=nl, 
        uniform_input_space=False)
    Y_val = scaler.transform(Y_val_original).copy()
    #Y_val = Y_val_original
    X_val = torch.tensor(X_val, dtype=torch.float32)
    Y_val_original = torch.tensor(Y_val_original, dtype=torch.float32)
    Y_val = torch.tensor(Y_val, dtype=torch.float32)

    data_valid = data_generator.ArtificialDataset(X_val, Y_val)
    validation_loader = torch.utils.data.DataLoader(
        data_valid, batch_size=BATCH_SIZE_LOADER,
        shuffle=False, num_workers=cpu_count)

    
    X_test, Y_test_original, Y_noisy = data_generator.data_4to8(
        N_test, noise_level=nl, 
        uniform_input_space=False, add_yfair=True)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_test_original = torch.tensor(Y_test_original, dtype=torch.float32)
    Y_noisy = torch.tensor(Y_noisy, dtype=torch.float32)

    data_test = data_generator.ArtificialDataset(X_test, Y_test_original)
    test_loader = torch.utils.data.DataLoader(
    data_test, batch_size=64,
    shuffle=False, num_workers=cpu_count)
    
    data_test_noisy = data_generator.ArtificialNoisyDataset(X_test, Y_noisy)
    test_noisy_loader = torch.utils.data.DataLoader(
    data_test_noisy, batch_size=64,
    shuffle=False, num_workers=cpu_count)
    
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
    
    model_name = 'weak_' + model_name
    
    if method_name == 'bnn':
        h = StrongVariationalNet(
            N_SAMPLES, input_size, output_size, PLV, dev).to(dev)
        #h = WeakVariationalNet(
        #    N_SAMPLES, input_size, output_size, PLV, dev).to(dev)
        #lr = lr*10

    elif method_name == 'ann':
        h = StandardNet(input_size, output_size).to(dev)
        #h = WeakStandardNet(input_size, output_size).to(dev)
        #lr = lr*10
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
                        scaler=scaler,
                        validation_loader=validation_loader,
                        OP=op_solver_dist,
                        dev=dev
                    )

    else:
        print('check method_learning variable')
        quit()


    model_used = train_NN.train(EPOCHS=EPOCHS)


    ##################################################################
    ##### Solving the Optimization Problem ###########################
    ##################################################################
    
    reg_result = []
    freg_result = []
    mse_result = []
        
    for M in M_SAMPLES:
        model_used.update_n_samples(n_samples=M)
     
        mse_loss = nn.MSELoss()
           
        # Construct the solver again for the optimization part
        op_solver = cnu.SolveConstrainedNewsvendor(params_t, 1, dev)
        op_solver_dist = cnu.SolveConstrainedNewsvendor(params_t, M, dev)
        op_solver_dist_noisy = cnu.SolveConstrainedNewsvendor(params_t, 32, dev)
        if not aleat_bool and method_name=='ann':
            op_solver_dist = op_solver
            model_used.update_n_samples(n_samples=1)
        
        f_total = 0
        f_total_noisy = 0
        f_total_best = 0
        mse_loss_result = 0
        n_batches = len(test_loader)
        

        for i, (tdata, tndata) in enumerate(zip(test_loader, test_noisy_loader)):
            
            x_test_batch, y_test_batch = tdata
            _, y_test_noisy_batch = tndata
            y_test_noisy_batch = torch.permute(y_test_noisy_batch, (1,0,2))
                
            x_test_batch = x_test_batch.to(dev)
            y_test_batch = y_test_batch.to(dev)
            y_test_noisy_batch = y_test_noisy_batch.to(dev)
            
            y_preds = model_used.forward_dist(x_test_batch, aleat_bool)
            y_preds = inverse_transform(y_preds)

            mse_loss_result += (mse_loss(
                y_preds.mean(axis=0), y_test_batch)/n_batches).detach()
            
            f_total += (op_solver_dist.end_loss_dist(
                y_preds, y_test_batch)/n_batches).detach()         
            
            f_total_noisy += (op_solver_dist_noisy.end_loss_dist(
                y_test_noisy_batch, y_test_batch)/n_batches).detach()
            
            f_total_best += (op_solver.cost_fn(
                y_test_batch.unsqueeze(0), y_test_batch)/n_batches).detach()
                        
        regret = f_total.item() - f_total_best.item()
        f_regret = f_total.item() - f_total_noisy.item()

        print('Results for seed = ', seed_number, 'and M = ', M)
        print('MSE loss: ', round(mse_loss_result.item(), 5))
        print('END cost: ', round(f_total.item(), 5))
        print('BEST cost: ', round(f_total_best.item(), 5))
        print('REGRET: ', round(regret, 5))
        print('FAIR REGRET: ', round(f_regret, 5))
        
        mse_result.append(mse_loss_result.item())
        reg_result.append(regret)
        freg_result.append(f_regret)
        
        
    return model_used, model_name, reg_result, freg_result, mse_result

if __name__ == '__main__':
    
    is_cuda = False
    dev = torch.device('cpu')  
    if torch.cuda.is_available():
        is_cuda = True
        dev = torch.device('cuda') 

        
    assert (len(sys.argv)==6)
    method_name = sys.argv[1] # ann or bnn
    method_learning = sys.argv[2] # decoupled or combined
    nr_seeds = int(sys.argv[3]) # Average results through seeds
    aleat_bool = bool(int(sys.argv[4])) # Modeling aleatoric uncert
    N_SAMPLES = int(sys.argv[5])  # Sampling size while training
    #M_SAMPLES = int(sys.argv[6])  # Sampling size while optimizing
    M_SAMPLES = [32, 16, 8, 4]
        
    mse_results_32 = []
    mse_results_16 = []
    mse_results_8 = []
    mse_results_4 = []
    
    reg_results_32 = []
    reg_results_16 = []
    reg_results_8 = []
    reg_results_4 = []
    
    freg_results_32 = []
    freg_results_16 = []
    freg_results_8 = []
    freg_results_4 = []
    
    for seed_number in range(0, nr_seeds):
        model_used, model_name, reg_result, freg_result, mse_result \
        = run_constrained_newsvendor(
            method_name, 
            method_learning,
            seed_number,
            aleat_bool,
            N_SAMPLES,
            M_SAMPLES,
            dev
        )
        
        mse_results_32.append(mse_result[0])
        mse_results_16.append(mse_result[1])
        mse_results_8.append(mse_result[2])
        mse_results_4.append(mse_result[3])
        
        reg_results_32.append(reg_result[0])
        reg_results_16.append(reg_result[1])
        reg_results_8.append(reg_result[2])
        reg_results_4.append(reg_result[3])
        
        freg_results_32.append(freg_result[0])
        freg_results_16.append(freg_result[1])
        freg_results_8.append(freg_result[2])
        freg_results_4.append(freg_result[3])
        
        
        ##########################################################
        ##### Saving model and results ###########################
        ##########################################################
    
        torch.save(model_used, f'./models/{model_name}.pkl')  
    
    df_total = pd.DataFrame(data={
        'MSE32':mse_results_32,
        'MSE16':mse_results_16, 
        'MSE8':mse_results_8, 
        'MSE4':mse_results_4,
        'REG32':reg_results_32,
        'REG16':reg_results_16,
        'REG8':reg_results_8,
        'REG4':reg_results_4,
        'FREG32':freg_results_32,
        'FREG16':freg_results_16,
        'FREG8':freg_results_8,
        'FREG4':freg_results_4
    })
    
    mse32_avg = df_total['MSE32'].mean()
    mse32_std = df_total['MSE32'].std()
    
    mse16_avg = df_total['MSE16'].mean()
    mse16_std = df_total['MSE16'].std()
    
    mse8_avg = df_total['MSE8'].mean()
    mse8_std = df_total['MSE8'].std()
    
    mse4_avg = df_total['MSE4'].mean()
    mse4_std = df_total['MSE4'].std()
    
    reg32_avg = df_total['REG32'].mean()
    reg32_std = df_total['REG32'].std()
    
    reg16_avg = df_total['REG16'].mean()
    reg16_std = df_total['REG16'].std()
    
    reg8_avg = df_total['REG8'].mean()
    reg8_std = df_total['REG8'].std()
    
    reg4_avg = df_total['REG4'].mean()
    reg4_std = df_total['REG4'].std()
    
    freg32_avg = df_total['FREG32'].mean()
    freg32_std = df_total['FREG32'].std()
    
    freg16_avg = df_total['FREG16'].mean()
    freg16_std = df_total['FREG16'].std()
    
    freg8_avg = df_total['FREG8'].mean()
    freg8_std = df_total['FREG8'].std()
    
    freg4_avg = df_total['FREG4'].mean()
    freg4_std = df_total['FREG4'].std()
    
    print('---------------------------------------------------')
    print('-----------------Results---------------------------')
    print(f'Params: {model_name}')
    print('MSE32: ', round(mse32_avg, 5), '(', round(mse32_std, 5), ')')
    print('MSE16: ', round(mse16_avg, 5), '(', round(mse16_std, 5), ')')
    print('MSE8: ', round(mse8_avg, 5), '(', round(mse8_std, 5), ')')
    print('MSE4: ', round(mse4_avg, 5), '(', round(mse4_std, 5), ')')
    print('REG32: ', round(reg32_avg, 5), '(', round(reg32_std, 5), ')')
    print('REG16: ', round(reg16_avg, 5), '(', round(reg16_std, 5), ')')
    print('REG8: ', round(reg8_avg, 5), '(', round(reg8_std, 5), ')')
    print('REG4: ', round(reg4_avg, 5), '(', round(reg4_std, 5), ')')
    print('FREG32: ', round(freg32_avg, 5), '(', round(freg32_std, 5), ')')
    print('FREG16: ', round(freg16_avg, 5), '(', round(freg16_std, 5), ')')
    print('FREG8: ', round(freg8_avg, 5), '(', round(freg8_std, 5), ')')
    print('FREG4: ', round(freg4_avg, 5), '(', round(freg4_std, 5), ')')
        
    df_total.to_csv(f'./newsvendor_results/{model_name}_nr.csv')