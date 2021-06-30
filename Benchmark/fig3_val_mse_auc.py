### Fig 3. Val MSE and performance code

import pandas as pd
import numpy as np
import pickle
import seaborn as sb

from scipy.stats import spearmanr, rankdata
from sklearn.linear_model import ElasticNetCV, ElasticNet

from sklearn.metrics import auc, mean_squared_error

from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

import random

from datasets import independent_data_linear, corr_group_data_linear, aml_data_linear
from models import MLP, XGBModel


###
### Initialize lists for results
###
xgb_mse_list = []
xgb_auc_list = []

mlp_mse_list = []
mlp_auc_list = []

enet_mse_list = []
enet_auc_list = []

##
## Generate a single dataset
##

X,y,beta = corr_group_data_multiplicative(0.99,0.0,0)
X_inner, X_outer, y_inner, y_outer = train_test_split(X,y)

for random_seed in range(20):
    print('Iteration {:d}'.format(random_seed))
    noise = 0.0
    
    ## bootstrap resample data
    X_new, y_new = resample(X_inner,y_inner)
    
    ###
    ### xgboost
    xgb_model = XGBModel()
    xgb_model.fit_grid_search(X_new,y_new)
    
    X_test_ss = xgb_model.feature_scaler.transform(X_outer)
    y_test_ss = xgb_model.outcome_scaler.transform(y_outer.reshape(-1,1))

    # calc metrics
    xgb_result = mean_squared_error(y_test_ss,xgb_model.best_model.predict(xgb.DMatrix(X_test_ss)))
    xgb_coefs = xgb_model.get_attribs()
    xgb_auc = get_auc(xgb_coefs,beta)
    xgb_mse_list.append(xgb_result)
    xgb_auc_list.append(xgb_auc)
    
    ###
    ### mlp
    mlp_model = MLP()
    mlp_model.fit_grid_search(X_new,y_new)
    
    X_test_ss = mlp_model.feature_scaler.transform(X_outer)
    y_test_ss = mlp_model.outcome_scaler.transform(y_outer.reshape(-1,1))

    # calc metrics
    mlp_result = mean_squared_error(y_test_ss,mlp_model.best_model(torch.tensor(X_test_ss).float()).detach())
    mlp_coefs = mlp_model.get_attribs()
    mlp_auc = get_auc(mlp_coefs,beta)
    mlp_mse_list.append(mlp_result)
    mlp_auc_list.append(mlp_auc)
    
    ###
    ### enet
    _, lm, linear_feature_scaler, linear_outcome_scaler = linear_MSE(X_new,y_new)
    
    X_test_ss = linear_feature_scaler.transform(X_outer)
    y_test_ss = linear_outcome_scaler.transform(y_outer.reshape(-1,1))

    # calc metrics
    enet_result = mean_squared_error(y_test_ss,lm.predict(X_test_ss))
    enet_coefs = np.abs(lm.coef_)
    enet_auc = get_auc(enet_coefs,beta)
    enet_mse_list.append(enet_result)
    enet_auc_list.append(enet_auc)
    
### save results    
results_frame = pd.DataFrame()
results_frame['xgb_mse'] = xgb_mse_list
results_frame['xgb_auc'] = xgb_auc_list
results_frame['mlp_mse'] = mlp_mse_list
results_frame['mlp_auc'] = mlp_auc_list
results_frame['enet_mse'] = enet_mse_list
results_frame['enet_auc'] = enet_auc_list
results_frame.to_csv('results/val_mse_feature_discovery_corr_group_multiplicative.csv')

### plot results
fig, ax1 = plt.subplots(1,1,figsize=(9,6))
ax1.scatter(np.array(mlp_mse_list),mlp_auc_list,color='#F39B7FFF')
ax1.scatter(np.array(xgb_mse_list),xgb_auc_list,color='#3C5488FF')
ax1.scatter(np.array(enet_mse_list),enet_auc_list,color='#4DBBD5FF')
# Hide the right and top spines
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)

for loc, spine in ax1.spines.items():
        if loc in ['left','bottom']:
            spine.set_position(('outward', 10))
plt.xlabel('MSE')
plt.ylabel('AUFDC')
plt.savefig('figures/val_mse_feature_discovery_corr_group_multiplicative.pdf')
plt.show()

