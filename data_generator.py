import numpy as np
import torch
from sklearn import preprocessing
import pdb

class ArtificialDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        return

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X_i = self.X[idx]
        y_i = self.y[idx]
        return X_i, y_i


def data_1to1(N, noise_level=1, noise_type='gaussian', uniform_input_space=False):
    
    if uniform_input_space:
        X = np.arange(-4, 4, 8/N)
    else:
        X = np.hstack((np.random.normal(-3, 1, N//2), np.random.normal(3, 1, N-N//2)))
    
    y_perfect = 5 + np.abs(np.abs(6*X)*np.sin(X) + np.sin(6*X)*(0.5*X))

    # Noise
    if noise_type == 'gaussian':
        n = np.where(X>3, 0, np.random.normal(0, noise_level, N))
        y = y_perfect + n*np.sin(X)*np.abs(X)*2
    elif noise_type == 'multimodal':
        n = np.hstack(
            (np.random.normal(6, noise_level, N//4), 
             np.random.normal(2, noise_level, N-N//4))
        )
        np.random.shuffle(n)
        n = np.where(X<3, n, 0)
        y = y_perfect + 0.5*n*np.sin(X)*np.abs(X)*2    
    elif noise_type == 'poisson':
        n = np.random.poisson(lam=5*noise_level, size=N)    
        np.random.shuffle(n)
        n = np.where(X<3, n, 0)
        y = y_perfect + 0.5*n*np.sin(X)*np.abs(X)*2      
    else:
        print('noise_type not considered')
        exit()
    
    y = np.where(y<0, 0, y)
    #X = np.expand_dims(preprocessing.scale(X), axis=1)
    #y = np.expand_dims(preprocessing.scale(y), axis=1)
    
    X = np.expand_dims(X, axis=1)
    y = np.expand_dims(y, axis=1)
    
    return X, y



def data_4to1(N, noise_level=1):

    x1 = np.random.normal(0, 1, size=N)
    x2 = np.random.poisson(lam=1.0, size=N)
    x3 = np.random.gamma(3, scale=1.0, size=N)
    x4 = np.random.uniform(low=-1.0, high=4.0, size=N)
    
    X = np.vstack((x1, x2, x3, x4)).T
    
    y_perfect = (
        3*x1*np.sin(x1*x2) 
        + x4*np.abs(x1*x3) 
        + 5*np.log(np.abs(x4)) 
        + 8*np.sin(3*x2*x4)
    )
    
    # Noise
    n = (
        np.random.normal(0, noise_level, N)*np.abs(x1*x2)
        + np.random.normal(0, noise_level, N)*x4*np.abs(x3)
    )
    y = y_perfect + n
    
    return X, y


def data_4to8(N, noise_level=1, seed_number=42):

    np.random.seed(seed_number)
    x1 = np.random.normal(0, 1, N)
    x2 = np.random.normal(0, 1, N)
    x3 = np.random.normal(0, 1, N)
    x4 = np.random.normal(0, 1, N)

    y1_perfect = np.maximum(5 + np.abs(6*x1)*np.sin(x2) + 4*np.sin(6*x3), 0)
    y2_perfect = np.maximum(3 + np.abs(10*x2)*np.sin(x3)**2 + 2*np.sin(6*x1), 0)
    y3_perfect = np.maximum(3 + np.abs(4*x3)**0.5*np.sin(x2) + 4*np.sin(2*x4), 0)
    y4_perfect = np.maximum(7 + np.abs(6*x4)*np.sin(x1) + 2*np.sin(6*x2)**2, 0)
    
    y5_perfect = y1_perfect + y2_perfect
    y6_perfect = y2_perfect + y3_perfect
    y7_perfect = y3_perfect + y4_perfect
    y8_perfect = y4_perfect + y1_perfect

    n_gaussian_1 = np.random.normal(1, noise_level, N)
    n_gaussian_2 = np.random.normal(1, 0.5*noise_level, N)
    n_multimodal_1 = np.hstack(
            (np.random.normal(0.5, noise_level, N//4), 
             np.random.normal(2, noise_level, N-N//4))
        )
    n_multimodal_2 = np.hstack(
            (np.random.normal(0.5, 0.5*noise_level, N//4), 
             np.random.normal(2, noise_level, N-N//4))
        )

    n_poisson_1 = np.random.poisson(lam=0.1*noise_level, size=N)
    n_poisson_2 = np.random.poisson(lam=0.2*noise_level, size=N)


    y1 = np.maximum(y1_perfect + n_gaussian_1, 0)
    y2 = np.maximum(y2_perfect + n_gaussian_2, 0)
    y3 = np.maximum(y3_perfect + n_multimodal_2, 0)
    y4 = np.maximum(y4_perfect + n_poisson_1, 0)
    
    y5 = np.maximum(y5_perfect + n_multimodal_1, 0)
    y6 = np.maximum(y6_perfect + n_gaussian_1, 0)
    y7 = np.maximum(y7_perfect + n_gaussian_2, 0)
    y8 = np.maximum(y8_perfect + n_poisson_2, 0)

    X = np.vstack((x1, x2, x3, x4)).T.round(3)
    Y = np.vstack((y1, y2, y3, y4, y5, y6, y7, y8)).T.round(3)
    
    return X, Y