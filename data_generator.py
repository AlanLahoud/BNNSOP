import numpy as np
import torch
from sklearn import preprocessing

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
    
class ArtificialNoisyDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        return

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X_i = self.X[idx]
        Y_i = self.Y[:,idx,:]
        return X_i, Y_i


# Creating a nonlinear (x, y) data
# with nonlinear relation and noise
def data_1to1(N, noise_level=1, seed_number=42, 
              noise_type='gaussian', 
              uniform_input_space=False, 
              add_yfair=False):
    
    np.random.seed(seed_number)
    if uniform_input_space:
        X = np.arange(-4, 4, 8/N)
    else:
        X = np.hstack((np.random.normal(-3, 1, N//2), 
                       np.random.normal(3, 1, N-N//2)))
    
    def gen_output(X):
        
        # Nonlinear relation between X and Y
        y_perfect = 5 + np.abs(
            np.abs(6*X)*np.sin(X) + np.sin(6*X)*(0.5*X))
        
        # Input dependent Noise
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
        y = np.expand_dims(y, axis=1)
        return y
    
    y = gen_output(X)
    
    if add_yfair:
        y_noisy = np.zeros((10000, N, 1))
        for i in range(0, 10000):
            y_noisy[i,:,:] = gen_output(X)
        y_noisy = torch.tensor(y_noisy, dtype=torch.float32)
    else:
        y_noisy = y
    
    X = np.expand_dims(X, axis=1)
    
    return X, y, y_noisy


# Creating a nonlinear (X, Y) data
# with nonlinear relation and noise
def data_4to8(N, noise_level=1, seed_number=42, 
              uniform_input_space=False, 
              add_yfair=False):

    np.random.seed(seed_number)
    
    if uniform_input_space:
        x1 = np.arange(-4, 4, 8/N)
        x2 = np.arange(-5, 5, 10/N)
        x3 = np.arange(-4, 4, 8/N)
        x4 = np.arange(-4, 3, 7/N)
    else:
        x1 = np.hstack((np.random.normal(-3, 1, N//2), 
                        np.random.normal(3, 1, N-N//2)))
        x2 = np.hstack((np.random.normal(-4, 1, N//2), 
                        np.random.normal(4, 1, N-N//2)))
        x3 = np.hstack((np.random.normal(-3, 0.7, N//2), 
                        np.random.normal(3, 0.7, N-N//2)))
        x4 = np.hstack((np.random.normal(-3, 1, N//2), 
                        np.random.normal(1, 2, N-N//2)))
    
    def gen_output(x1, x2, x3, x4):
        y1_perfect = np.maximum(
            10 + np.abs(x1)*np.sin(x2) + 4*np.sin(6*x3), 0)
        y2_perfect = np.maximum(
            3 + np.abs(10*x2)*np.sin(x3)**2 + 2*np.sin(6*x1), 0)
        y3_perfect = np.maximum(
            10 + np.abs(4*x3)**0.5*np.sin(x2) + 4*np.sin(2*x4), 0)
        y4_perfect = np.maximum(
            7 + np.abs(6*x4)*np.sin(x1) + 2*np.sin(6*x2)**2, 0)

        y5_perfect = 5 + y1_perfect + y2_perfect
        y6_perfect = 5 + y2_perfect + y3_perfect

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

        n_poisson_1 = np.random.poisson(
            lam=0.2*noise_level, size=N)
        n_poisson_2 = np.random.poisson(
            lam=0.4*noise_level, size=N)


        y1 = np.maximum(y1_perfect + n_gaussian_1, 0)
        y2 = np.maximum(y2_perfect + n_gaussian_2, 0)
        y3 = np.maximum(y3_perfect + n_multimodal_2, 0)
        y4 = np.maximum(y4_perfect + n_poisson_1, 0)

        y5 = np.maximum(y5_perfect + n_multimodal_1, 0)
        y6 = np.maximum(y6_perfect + n_gaussian_1, 0)

        Y = np.vstack((y1, y2, y3, y4, y5, y6)).T.round(3)
        return Y
    
    Y = gen_output(x1, x2, x3, x4)

    np.random.seed(0)
    if add_yfair:
        Y_noisy = np.zeros((32, N, 6))
        for i in range(0, 32):
            Y_noisy[i,:,:] = gen_output(x1, x2, x3, x4)
        Y_noisy = torch.tensor(Y_noisy, dtype=torch.float32)
    else:
        Y_noisy = Y

    X = np.vstack((x1, x2, x3, x4)).T.round(3)
    
    return X, Y, Y_noisy




# Data for portfolio risk optimization (minimax)
def gen_intermediate(num, n_assets, x1, x2, x3):      
        factor = num * 2/(n_assets)
        return x1**factor + x2**factor + x3**factor
    
def gen_data(N, n_assets, nl, seed_number=42, samples_dist=1):
    np.random.seed(seed_number)
    x1 = np.random.normal(1, 1, size = N).clip(0)
    x2 = np.random.normal(1, 1, size = N).clip(0)
    x3 = np.random.normal(1, 1, size = N).clip(0)
    X = np.vstack((x1, x2, x3)).T
    
    def geny(x1, x2, x3, N, n_assets):
        Y = np.zeros((N, n_assets))
        for i in range(1, n_assets + 1):
            interm = gen_intermediate(i, n_assets, x1, x2, x3)
            Y[:,i-1] = (np.sin(interm) - np.sin(interm).mean())

            #Y[:,i-1] = 0.7 + Y[:,i-1] - np.abs(nl*np.abs(x1 + x2)*(np.random.exponential(1, size=Y[:,0].shape)))
                        
            Y[:,i-1] = 0.2 + Y[:,i-1] - np.abs(nl*np.abs(x1)*(np.random.lognormal(0, 0.7, size=Y[:,0].shape)))
                        
        return Y
        
    Y = geny(x1, x2, x3, N, n_assets)
        
    Y_dist = np.zeros((samples_dist, N, n_assets))
    for i in range(0, samples_dist):
        Y_dist[i, :, :] = geny(x1, x2, x3, N, n_assets)
        
    return X, Y, Y_dist

def gen_cond_dist(N, n_assets, n_samples, nl, seed_number=420):
    np.random.seed(seed_number)
    Y_dist = np.zeros((n_samples, N, n_assets))
    for i in range(0, n_samples):
        Y_dist[i, :, :] = gen_data(N, n_assets, nl, seed_number=np.random.randint(0,999999))[1]
    return Y_dist