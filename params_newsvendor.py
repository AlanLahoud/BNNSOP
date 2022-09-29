import pandas as pd
import numpy as np
import torch

#################################################################################
## OP deterministic parameters ##################################################
#################################################################################

def get_params(n_items, seed_number):
    
    np.random.seed(seed_number)
    
    params_name = ['q','qs','qw','c','cs','cw','pr','si']
    params_list = []
    
    for i in range(0, n_items):
        
        q = 6 + np.random.randint(-3, 3) # Fixed cost for each item
        qs = 20 + np.random.randint(-5, 5)  # Shortage cost fpr each item
        qw = 15 + np.random.randint(-5, 5) # Excess cost for each item
        
        c = 30 + np.random.randint(-5, 5) # Fixed cost for each item
        cs = 240 + np.random.randint(-50, 50)  # Shortage cost fpr each item
        cw = 100 + np.random.randint(-20, 20) # Excess cost for each item
        
        # constraints of price and size for each item
        pr = int(100000/(c + cs + cw) + np.random.randint(-50, 50))
        si = int(100000/(c + cs + cw) + np.random.randint(-50, 50))

        params_list.append([q, qs, qw, c, cs, cw, pr, si])

    df_parameters = pd.DataFrame(data=params_list, columns = params_name)

    # Generate a bound for inequalities of Budget and Size
    avg_sales = 7 
    B = 400*avg_sales*n_items*np.random.uniform(0.3, 0.6)
    S = 400*avg_sales*n_items*np.random.uniform(0.3, 0.6)
    
    
    # Building the parameters as numpy and torch dictionary
    params = {}
    
    params['q'] = df_parameters['q'].tolist()
    params['qs'] = df_parameters['qs'].tolist()
    params['qw'] = df_parameters['qw'].tolist()
    
    params['c'] = df_parameters['c'].tolist()
    params['cs'] = df_parameters['cs'].tolist()
    params['cw'] = df_parameters['cw'].tolist()
    
    params['pr'] = df_parameters['pr'].tolist()
    params['B'] = [B]
    
    params['si'] = df_parameters['si'].tolist()
    params['S'] = [S]

    for key in params.keys():
        params[key] = torch.Tensor(params[key])
        params_t = params.copy()
        
    for key in params.keys():
        params[key] = np.array(params[key])
        params_np = params.copy()

    return params_t, params_np