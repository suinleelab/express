import numpy as np
import pandas as pd
import random

##
## Linear datasets
##

def independent_data_linear(noise,seed):
    np.random.seed(seed)
    random.seed(seed)

    # generate dataset with known correlation
    N = 1000
    M = 100

    # pick 10 coefficients at random to be non-zero
    rand_10 = random.sample(range(M),10)
    beta = np.zeros(M)
    beta[rand_10] = 1

    f = lambda X: np.matmul(X, beta)

    # create the final data
    X = np.random.randn(N, M)
    y = f(X) + noise*np.random.randn(N)
    
    return X,y,beta

def corr_group_data_linear(corr,noise,seed):
    np.random.seed(seed)
    random.seed(seed)

    # generate dataset with known correlation
    N = 1000
    M = 100

    # pick 10 coefficients at random to be non-zero
    rand_10 = random.sample(range(M),10)
    beta = np.zeros(M)
    beta[rand_10] = 1

    # build a correlation matrix with 10 groups of 5 tightly correlated features
    C = np.eye(M)
    for i in range(0,M,10):
        C[i,i+1] = C[i+1,i] = corr
        C[i,i+2] = C[i+2,i] = corr
        C[i,i+3] = C[i+3,i] = corr
        C[i,i+4] = C[i+4,i] = corr
        C[i+1,i+2] = C[i+2,i+1] = corr
        C[i+1,i+3] = C[i+3,i+1] = corr
        C[i+1,i+4] = C[i+4,i+1] = corr
        C[i+2,i+3] = C[i+3,i+2] = corr
        C[i+2,i+4] = C[i+4,i+2] = corr
        C[i+3,i+4] = C[i+4,i+3] = corr
    f = lambda X: np.matmul(X, beta)

    # Make sure the sample correlation is a perfect match
    X_start = np.random.randn(N, M)
    X_centered = X_start - X_start.mean(0)
    Sigma = np.matmul(X_centered.T, X_centered) / X_centered.shape[0]
    W = np.linalg.cholesky(np.linalg.inv(Sigma)).T
    X_white = np.matmul(X_centered, W.T)
    assert np.linalg.norm(np.corrcoef(np.matmul(X_centered, W.T).T) - np.eye(M)) < 1e-6 # ensure this decorrelates the data

    # create the final data
    X_final = np.matmul(X_white, np.linalg.cholesky(C).T)
    X = X_final
    y = f(X) + noise*np.random.randn(N)
    
    return X,y,beta

def aml_data_linear(noise,seed):
    np.random.seed(seed)
    random.seed(seed)
    rna_seq = pd.read_csv('data/semi_synthetic_rna_seq.csv',index_col=0).sample(100,axis=1,random_state=seed)
    rna_seq = rna_seq.loc[~rna_seq.duplicated(subset=None, keep='first'),:]
    
    X = rna_seq.values
    # generate dataset with known correlation
    N = X.shape[0]
    M = 100

    # pick 10 coefficients at random to be non-zero
    rand_10 = random.sample(range(M),10)
    beta = np.zeros(M)
    beta[rand_10] = 1

    f = lambda X: np.matmul(X, beta)
    
    # create the final data
    y = f(X) + noise*np.random.randn(N)
    
    return X,y,beta

##
## AND datasets
##

def independent_data_AND(noise,seed):
    np.random.seed(seed)
    random.seed(seed)

    # generate dataset with known correlation
    N = 1000
    M = 100

    # pick 10 coefficients at random to be non-zero
    rand_10 = random.sample(range(M),10)
    beta = np.zeros(M)
    beta[rand_10] = 1

    # create the final data
    X = np.random.randn(N, M)
    y = noise*np.random.randn(N)
    for key in range(9):
        y += np.logical_and(X[:,rand_10[key]] > 0, X[:,rand_10[key+1]] < 0).astype(int)
    
    return X,y,beta

def corr_group_data_AND(corr,noise,seed):
    np.random.seed(seed)
    random.seed(seed)

    # generate dataset with known correlation
    N = 1000
    M = 100

    # pick 10 coefficients at random to be non-zero
    rand_10 = random.sample(range(M),10)
    beta = np.zeros(M)
    beta[rand_10] = 1

    # build a correlation matrix with 10 groups of 5 tightly correlated features
    C = np.eye(M)
    for i in range(0,M,10):
        C[i,i+1] = C[i+1,i] = corr
        C[i,i+2] = C[i+2,i] = corr
        C[i,i+3] = C[i+3,i] = corr
        C[i,i+4] = C[i+4,i] = corr
        C[i+1,i+2] = C[i+2,i+1] = corr
        C[i+1,i+3] = C[i+3,i+1] = corr
        C[i+1,i+4] = C[i+4,i+1] = corr
        C[i+2,i+3] = C[i+3,i+2] = corr
        C[i+2,i+4] = C[i+4,i+2] = corr
        C[i+3,i+4] = C[i+4,i+3] = corr

    # Make sure the sample correlation is a perfect match
    X_start = np.random.randn(N, M)
    X_centered = X_start - X_start.mean(0)
    Sigma = np.matmul(X_centered.T, X_centered) / X_centered.shape[0]
    W = np.linalg.cholesky(np.linalg.inv(Sigma)).T
    X_white = np.matmul(X_centered, W.T)
    assert np.linalg.norm(np.corrcoef(np.matmul(X_centered, W.T).T) - np.eye(M)) < 1e-6 # ensure this decorrelates the data

    # create the final data
    X_final = np.matmul(X_white, np.linalg.cholesky(C).T)
    X = X_final
    y = noise*np.random.randn(N)
    for key in range(9):
        y += np.logical_and(X[:,rand_10[key]] > 0, X[:,rand_10[key+1]] < 0).astype(int)
    
    return X,y,beta

def aml_data_AND(noise,seed):
    np.random.seed(seed)
    random.seed(seed)
    rna_seq = pd.read_csv('data/semi_synthetic_rna_seq.csv',index_col=0).sample(100,axis=1,random_state=seed)
    rna_seq = rna_seq.loc[~rna_seq.duplicated(subset=None, keep='first'),:]
    
    X = rna_seq.values
    # generate dataset with known correlation
    N = X.shape[0]
    M = 100

    # pick 10 coefficients at random to be non-zero
    rand_10 = random.sample(range(M),10)
    beta = np.zeros(M)
    beta[rand_10] = 1
    
    # create the final data
    y = noise*np.random.randn(N)
    for key in range(9):
        y += np.logical_and(X[:,rand_10[key]] > 0, X[:,rand_10[key+1]] < 0).astype(int)
    
    return X,y,beta

##
## Multiplicative datasets
##

def independent_data_multiplicative(noise,seed):
    np.random.seed(seed)
    random.seed(seed)

    # generate dataset with known correlation
    N = 1000
    M = 100

    # pick 10 coefficients at random to be non-zero
    rand_10 = random.sample(range(M),10)
    beta = np.zeros(M)
    beta[rand_10] = 1

    # create the final data
    X = np.random.randn(N, M)
    y = noise*np.random.randn(N)
    for key in range(9):
        y += X[:,rand_10[key]]*X[:,rand_10[key+1]]
    
    return X,y,beta

def corr_group_data_multiplicative(corr,noise,seed):
    np.random.seed(seed)
    random.seed(seed)

    # generate dataset with known correlation
    N = 1000
    M = 100

    # pick 10 coefficients at random to be non-zero
    rand_10 = random.sample(range(M),10)
    beta = np.zeros(M)
    beta[rand_10] = 1

    # build a correlation matrix with 10 groups of 5 tightly correlated features
    C = np.eye(M)
    for i in range(0,M,10):
        C[i,i+1] = C[i+1,i] = corr
        C[i,i+2] = C[i+2,i] = corr
        C[i,i+3] = C[i+3,i] = corr
        C[i,i+4] = C[i+4,i] = corr
        C[i+1,i+2] = C[i+2,i+1] = corr
        C[i+1,i+3] = C[i+3,i+1] = corr
        C[i+1,i+4] = C[i+4,i+1] = corr
        C[i+2,i+3] = C[i+3,i+2] = corr
        C[i+2,i+4] = C[i+4,i+2] = corr
        C[i+3,i+4] = C[i+4,i+3] = corr

    # Make sure the sample correlation is a perfect match
    X_start = np.random.randn(N, M)
    X_centered = X_start - X_start.mean(0)
    Sigma = np.matmul(X_centered.T, X_centered) / X_centered.shape[0]
    W = np.linalg.cholesky(np.linalg.inv(Sigma)).T
    X_white = np.matmul(X_centered, W.T)
    assert np.linalg.norm(np.corrcoef(np.matmul(X_centered, W.T).T) - np.eye(M)) < 1e-6 # ensure this decorrelates the data

    # create the final data
    X_final = np.matmul(X_white, np.linalg.cholesky(C).T)
    X = X_final
    y = noise*np.random.randn(N)
    for key in range(9):
        y += X[:,rand_10[key]]*X[:,rand_10[key+1]]
    
    return X,y,beta

def aml_data_multiplicative(noise,seed):
    np.random.seed(seed)
    random.seed(seed)
    rna_seq = pd.read_csv('data/semi_synthetic_rna_seq.csv',index_col=0).sample(100,axis=1,random_state=seed)
    rna_seq = rna_seq.loc[~rna_seq.duplicated(subset=None, keep='first'),:]
    
    X = rna_seq.values
    # generate dataset with known correlation
    N = X.shape[0]
    M = 100

    # pick 10 coefficients at random to be non-zero
    rand_10 = random.sample(range(M),10)
    beta = np.zeros(M)
    beta[rand_10] = 1
    
    # create the final data
    y = noise*np.random.randn(N)
    for key in range(9):
        y += X[:,rand_10[key]]*X[:,rand_10[key+1]]
    
    return X,y,beta

##
## ReLU datasets
##

def independent_data_relu(noise,seed):
    np.random.seed(seed)
    random.seed(seed)

    # generate dataset with known correlation
    N = 1000
    M = 100

    # pick 10 coefficients at random to be non-zero
    rand_10 = random.sample(range(M),10)
    beta = np.zeros(M)
    beta[rand_10] = 1

    # create the final data
    X = np.random.randn(N, M)
    y = noise*np.random.randn(N)
    for key in range(9):
        y += np.maximum(X[:,rand_10[key]]+X[:,rand_10[key+1]],0)
    
    return X,y,beta

def corr_group_data_relu(corr,noise,seed):
    np.random.seed(seed)
    random.seed(seed)

    # generate dataset with known correlation
    N = 1000
    M = 100

    # pick 10 coefficients at random to be non-zero
    rand_10 = random.sample(range(M),10)
    beta = np.zeros(M)
    beta[rand_10] = 1

    # build a correlation matrix with 10 groups of 5 tightly correlated features
    C = np.eye(M)
    for i in range(0,M,10):
        C[i,i+1] = C[i+1,i] = corr
        C[i,i+2] = C[i+2,i] = corr
        C[i,i+3] = C[i+3,i] = corr
        C[i,i+4] = C[i+4,i] = corr
        C[i+1,i+2] = C[i+2,i+1] = corr
        C[i+1,i+3] = C[i+3,i+1] = corr
        C[i+1,i+4] = C[i+4,i+1] = corr
        C[i+2,i+3] = C[i+3,i+2] = corr
        C[i+2,i+4] = C[i+4,i+2] = corr
        C[i+3,i+4] = C[i+4,i+3] = corr

    # Make sure the sample correlation is a perfect match
    X_start = np.random.randn(N, M)
    X_centered = X_start - X_start.mean(0)
    Sigma = np.matmul(X_centered.T, X_centered) / X_centered.shape[0]
    W = np.linalg.cholesky(np.linalg.inv(Sigma)).T
    X_white = np.matmul(X_centered, W.T)
    assert np.linalg.norm(np.corrcoef(np.matmul(X_centered, W.T).T) - np.eye(M)) < 1e-6 # ensure this decorrelates the data

    # create the final data
    X_final = np.matmul(X_white, np.linalg.cholesky(C).T)
    X = X_final
    y = noise*np.random.randn(N)
    for key in range(9):
        y += np.maximum(X[:,rand_10[key]]+X[:,rand_10[key+1]],0)
    
    return X,y,beta

def aml_data_relu(noise,seed):
    np.random.seed(seed)
    random.seed(seed)
    rna_seq = pd.read_csv('data/semi_synthetic_rna_seq.csv',index_col=0).sample(100,axis=1,random_state=seed)
    rna_seq = rna_seq.loc[~rna_seq.duplicated(subset=None, keep='first'),:]
    
    X = rna_seq.values
    # generate dataset with known correlation
    N = X.shape[0]
    M = 100

    # pick 10 coefficients at random to be non-zero
    rand_10 = random.sample(range(M),10)
    beta = np.zeros(M)
    beta[rand_10] = 1
    
    # create the final data
    y = noise*np.random.randn(N)
    for key in range(9):
        y += np.maximum(X[:,rand_10[key]]+X[:,rand_10[key+1]],0)
    
    return X,y,beta