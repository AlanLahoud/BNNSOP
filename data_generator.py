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


def data_4to4(N, noise_level=1):

    x1 = np.random.normal(0, 1, size=N)
    x2 = np.random.poisson(lam=1.0, size=N)
    x3 = np.random.gamma(3, scale=1.0, size=N)
    x4 = np.random.uniform(low=-1.0, high=4.0, size=N)
    
    X = np.vstack((x1, x2, x3, x4)).T
    
    y1_perfect = np.abs((
        3*x1*np.sin(x1*x2) 
        + x4*np.abs(x1*x3) 
        + 5*np.log(0.1+np.abs(x4)) 
        + 8*np.sin(3*x2*x4)
    ))
    
    y2_perfect = np.abs((
        3*x2*np.sin(x1*x3) 
        + x4*np.abs(x2) 
        + 5*np.log(0.1+np.abs(x4)) 
        + 8*np.sin(3*x2*x3)
    ))
    
    y3_perfect = np.abs((
        5*x3*np.sin(x4) 
        + x4*np.abs(x3) 
        + 1*np.log(0.1+np.abs(x2)) 
        + 2*np.sin(x1)
    ))
    
    y4_perfect = np.abs((
        2*x4*np.sin(x4) 
        + x2*np.abs(x1) 
        + 1*np.log(0.1+np.abs(x2)) 
        + 12*np.sin(x4)
    ))
    
    # Noise
    def random_noise_():
        return (
            np.random.normal(0, noise_level, N)*np.abs(x1*x2)
            + np.random.normal(0, noise_level, N)*x4*np.abs(x3)
        )
    
    y1 = y1_perfect + random_noise_()
    y2 = y2_perfect + random_noise_()
    y3 = y3_perfect + random_noise_()
    y4 = y4_perfect + random_noise_()
    
    Y = np.vstack((y1, y2, y3, y4)).T
    
    
    return X, Y