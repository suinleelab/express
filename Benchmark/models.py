import torch
from torch.nn import Module
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, BatchSampler
import torch.optim as optim
from copy import deepcopy

from itertools import product

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import sage
import xgboost as xgb

import numpy as np

class XGBModel(object):
    'A simple XGBoost model for feature discovery'
    def __init__(self):
        '''
        Create an XGBoost for biological discovery.
        '''
        self.X_t = None
        self.y_t = None
        self.X_v = None
        self.y_v = None
        
    def fit_grid_search(self, X, y, verbose=False):
        '''
        Do a quick grid search on a small set of HPs and pick an optimal
        set for downstream experiments.
        '''
        
        X_train, X_val, y_train, y_val = train_test_split(X,y)
        feature_scaler = StandardScaler()
        outcome_scaler = StandardScaler()

        y_train = y_train.reshape(-1, 1)
        y_val = y_val.reshape(-1, 1)

        feature_scaler.fit(X_train)
        outcome_scaler.fit(y_train)
        
        self.feature_scaler = feature_scaler
        self.outcome_scaler = outcome_scaler

        X_train_ss,X_val_ss = feature_scaler.transform(X_train),feature_scaler.transform(X_val)
        y_train_ss,y_val_ss = outcome_scaler.transform(y_train),outcome_scaler.transform(y_val)
        
        self.X_t = X_train_ss
        self.y_t = y_train_ss
        self.X_v = X_val_ss
        self.y_v = y_val_ss
        
        max_depth = [x for x in range(2,50,8)]
        eta = [.3, .2, .1, .05, .01, .005]
        
        self.external_criterion = np.inf
        self.best_params = None
        self.best_model = None
        
        # grid search
        for params in product(max_depth, eta):
            ## build model here
            params_dict = {'max_depth' : params[0],
                           'eta' : params[1],
                           'objective': 'reg:squarederror'}
            current_model, crit = self._fit_model(params_dict, X_train_ss, X_val_ss, y_train_ss, y_val_ss,
                                                  verbose=verbose)
            if crit < self.external_criterion:
                self.best_params = params
                self.best_model = current_model
                self.external_criterion = crit
                
                
    def _fit_model(self, p_dict, X_train, X_val, y_train, y_val, verbose=False):
        # Training parameters
        num_round = 1000
        
        # Data loaders
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        # Setup
        evallist = [(dval, 'val')]

        # Train
        model = xgb.train(p_dict, dtrain, num_round, evallist, verbose_eval=verbose)
        min_criterion = mean_squared_error(model.predict(dval, ntree_limit=model.best_ntree_limit), y_val)
        
        return model, min_criterion
    
    def get_attribs(self, background_imputer='marginal'):
        # Setup
        f = lambda x: self.best_model.predict(xgb.DMatrix(x), ntree_limit=self.best_model.best_ntree_limit)
        if background_imputer == 'marginal':
            imputer = sage.utils.MarginalImputer(self.X_t, 512)
        if background_imputer == 'mvn':
            imputer = sage.utils.ConditionalImputer(self.X_t, 512)
        sampler = sage.PermutationSampler(f, imputer, 'mse')

        # Calculate SAGE values
        sage_values = sampler(xy=(self.X_v, self.y_v),
                              batch_size=2 ** 6,
                              n_permutations=2 ** 10,
                              bar=True,
                              verbose=True)
        
        return sage_values.values

class MLP(object):
    'A simple MLP for feature discovery'
    def __init__(self):
        '''
        Create an MLP for biological discovery.
        '''
        self.lossfunc = nn.MSELoss()
        self.X_t = None
        self.y_t = None
        self.X_v = None
        self.y_v = None
    
    def fit_grid_search(self, X, y, verbose=False):
        '''
        Do a quick grid search on a small set of HPs and pick an optimal
        set for downstream experiments.
        '''
        
        X_train, X_val, y_train, y_val = train_test_split(X,y)
        feature_scaler = StandardScaler()
        outcome_scaler = StandardScaler()

        y_train = y_train.reshape(-1, 1)
        y_val = y_val.reshape(-1, 1)

        feature_scaler.fit(X_train)
        outcome_scaler.fit(y_train)
        
        self.feature_scaler = feature_scaler
        self.outcome_scaler = outcome_scaler

        X_train_ss,X_val_ss = feature_scaler.transform(X_train),feature_scaler.transform(X_val)
        y_train_ss,y_val_ss = outcome_scaler.transform(y_train),outcome_scaler.transform(y_val)
        
        self.X_t = X_train_ss
        self.y_t = y_train_ss
        self.X_v = X_val_ss
        self.y_v = y_val_ss
        
        n_layers = [2,3,4]
        activation = ['ELU','ReLU']
        n_nodes = [64,128,256]
        shape = ['decreasing', 'non-decreasing']
        
        self.external_criterion = np.inf
        self.best_params = None
        self.best_model = None
        
        # grid search
        for params in product(n_layers, activation,
                              n_nodes, shape):
            current_model = self._build_model(params)
            current_model, crit = self._fit_model(current_model, X_train_ss, X_val_ss, y_train_ss, y_val_ss,
                                                  verbose=verbose)
            if crit < self.external_criterion:
                self.best_params = params
                self.best_model = current_model
                self.external_criterion = crit
            

    def _build_model(self, params):
        n_layers = params[0]
        activation = params[1]
        n_nodes = params[2]
        shape = params[3]
        
        architecture = []
        if shape == 'decreasing':
            current_nodes = n_nodes
            for i in range(n_layers):
                if i == 0:
                    architecture.append(nn.Linear(self.X_t.shape[1], current_nodes))
                    if activation == 'ELU':
                        architecture.append(nn.ELU())
                    else:
                        architecture.append(nn.ReLU())
                    last_nodes = int(current_nodes)
                    current_nodes = int(last_nodes / 2)
                elif i == (n_layers - 1):
                    architecture.append(nn.Linear(last_nodes,1))
                else:
                    architecture.append(nn.Linear(last_nodes, current_nodes))
                    if activation == 'ELU':
                        architecture.append(nn.ELU())
                    else:
                        architecture.append(nn.ReLU())
                    last_nodes = int(current_nodes)
                    current_nodes = int(last_nodes / 2)
        else:
            for i in range(n_layers):
                if i == 0:
                    architecture.append(nn.Linear(self.X_t.shape[1], n_nodes))
                    if activation == 'ELU':
                        architecture.append(nn.ELU())
                    else:
                        architecture.append(nn.ReLU())
                elif i == (n_layers - 1):
                    architecture.append(nn.Linear(n_nodes,1))
                else:
                    architecture.append(nn.Linear(n_nodes, n_nodes))
                    if activation == 'ELU':
                        architecture.append(nn.ELU())
                    else:
                        architecture.append(nn.ReLU())
        return nn.Sequential(*architecture)
        
    def _fit_model(self, model, X_train, X_val, y_train, y_val, verbose=False):
        # Training parameters
        lr = 1e-3
        mbsize = 64
        max_nepochs = 1000
        loss_fn = self.lossfunc
        lookback = 50
#         verbose = True
        
        # Data loaders
        train_set = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32))
        train_loader = DataLoader(train_set, batch_size=mbsize, shuffle=True)
        val_x = torch.tensor(X_val, dtype=torch.float32)
        val_y = torch.tensor(y_val, dtype=torch.float32)

        # Setup
        optimizer = optim.Adam(model.parameters(), lr=lr)
        min_criterion = np.inf
        min_epoch = 0

        # Train
        for epoch in range(max_nepochs):
            for x, y in train_loader:
                # Move to device.
                x = x
                y = y

                # Take gradient step.
                loss = loss_fn(model(x), y)
                loss.backward()
                optimizer.step()
                model.zero_grad()

            # Check progress.
            with torch.no_grad():
                # Calculate validation loss.
                val_loss = loss_fn(model(val_x), val_y).item()
                if verbose:
                    print('{}Epoch = {}{}'.format('-' * 10, epoch + 1, '-' * 10))
                    print('Val loss = {:.4f}'.format(val_loss))

                # Check convergence criterion.
                if val_loss < min_criterion:
                    min_criterion = val_loss
                    min_epoch = epoch
                    best_model = deepcopy(model)
                elif (epoch - min_epoch) == lookback:
                    if verbose:
                        print('Stopping early')
                    break

        # Keep best model
        return best_model, min_criterion
    
    def get_attribs(self, background_imputer='marginal'):
        # Setup
        f = lambda x: self.best_model(torch.tensor(x, dtype=torch.float32)).data.numpy()
        if background_imputer == 'marginal':
            imputer = sage.utils.MarginalImputer(self.X_t, 512)
        if background_imputer == 'mvn':
            imputer = sage.utils.ConditionalImputer(self.X_t, 512)
        sampler = sage.PermutationSampler(f, imputer, 'mse')

        # Calculate SAGE values
        sage_values = sampler(xy=(self.X_v, self.y_v),
                              batch_size=2 ** 6,
                              n_permutations=2 ** 10,
                              bar=True,
                              verbose=True)
        
        return sage_values.values