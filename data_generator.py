import numpy as np


def data_1to1(N, xl=-1, xh=1, noise_level=1):
    X = np.arange(xl, xh, (xh-xl)/N)
    
    y_perfect = np.abs(6*X)*np.sin(X) + np.sin(12*X)

    # Noise
    n = np.random.normal(0, noise_level, N)*np.abs(X**2)
    y = y_perfect + n
    
    X = np.expand_dims(X, axis=1)
    
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