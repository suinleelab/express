# fig 3 ensemble experiment

import pandas as pd
import numpy as np
import pickle
import seaborn as sb

from scipy.stats import spearmanr, rankdata
from sklearn.linear_model import ElasticNetCV, ElasticNet

from sklearn.metrics import auc

from sklearn.feature_selection import RFE
from sklearn.svm import SVR

import random

from shap import LinearExplainer

from datasets import corr_group_data_AND

from models import MLP, XGBModel

def linear_coefs_CV(X,y):
    lm = ElasticNetCV(cv=5,l1_ratio=0.99)
    lm.fit(X,y)
    return np.abs(lm.coef_)

def get_mean_ranks(feature_importance,beta):
    return np.mean((101 - rankdata(feature_importance))[np.argwhere(beta == 1)])

def get_auc(feature_importance,beta):
    running_total = [0]
    for x in np.argsort(feature_importance)[::-1]:
        if x in np.where(beta == 1)[0]:
            running_total.append(running_total[-1] + 1)
        else:
            running_total.append(running_total[-1])
    return auc(np.arange(len(running_total)),running_total)

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import xgboost as xgb
from sklearn.utils import resample

def ensemble_experiment():

    noise = 0.00
    random_seed = 0
    X,y,beta = corr_group_data_AND(0.99,noise,random_seed)

    method = []
    dataset = []
    noise_list = []
    auc_list = []
    mean_rank = []
    conditional_list = []
    mse_list = []
    
    ensemble_method = []
    ensemble_dataset = []
    ensemble_noise_list = []
    ensemble_auc_list = []
    ensemble_mean_rank = []
    ensemble_conditional_list = []
    ensemble_mse_list = []
    
    for ensemble_loop in range(20):
        
        shaps_list_xgb_int = []
        shaps_list_xgb_obs = []
        shaps_list_enet_int = []
        shaps_list_enet_obs = []

        for random_seed in range(20):
            print('Iteration {:d}'.format(random_seed))
            noise = 0.00
            X_new,y_new = resample(X,y)

            X_train, X_val, y_train, y_val = train_test_split(X_new,y_new)


            ###
            ### linear model
            lm = ElasticNetCV(cv=5,l1_ratio=0.99)
            lm.fit(X_train,y_train)
            conditional_exp = LinearExplainer(lm,X_train,nsamples=500,feature_dependence="independent")
            conditional_shaps = conditional_exp.shap_values(X_train)

            enet_feature_attrib = np.mean(np.abs(conditional_shaps),axis=0)

            # calc metrics
            enet_auc = get_auc(enet_feature_attrib,beta)
            enet_mean_rank = get_mean_ranks(enet_feature_attrib,beta)

            method.append('elastic net')
            dataset.append('aml')
            noise_list.append(noise)
            auc_list.append(enet_auc)
            mean_rank.append(enet_mean_rank)
            conditional_list.append('interventional')
            mse_list.append(mean_squared_error(lm.predict(X_val),y_val))

            shaps_list_enet_int.append(enet_feature_attrib)


            ###
            ### xgboost
            xgb_model = XGBModel()
            xgb_model.fit_grid_search(X_new,y_new)
            xgb_coefs = xgb_model.get_attribs()

            # calc metrics
            xgb_auc = get_auc(xgb_coefs,beta)
            xgb_mean_rank = get_mean_ranks(xgb_coefs,beta)

            method.append('xgboost')
            dataset.append('aml')
            noise_list.append(noise)
            auc_list.append(xgb_auc)
            mean_rank.append(xgb_mean_rank)
            conditional_list.append('interventional')
            shaps_list_xgb_int.append(xgb_coefs)
            mse_list.append(mean_squared_error(xgb_model.best_model.predict(xgb.DMatrix(xgb_model.X_v)),xgb_model.y_v))
        
        ensemble_shaps_xgb_int = np.zeros_like(shaps_list_xgb_int[0])
        for x in shaps_list_xgb_int:
            ensemble_shaps_xgb_int += x
        ensemble_xgb_auc = get_auc(ensemble_shaps_xgb_int,beta)
        ensemble_xgb_mean_rank = get_mean_ranks(ensemble_shaps_xgb_int,beta)
        ensemble_method.append('xgboost')
        ensemble_dataset.append('aml')
        ensemble_noise_list.append(noise)
        ensemble_auc_list.append(ensemble_xgb_auc)
        ensemble_mean_rank.append(ensemble_xgb_mean_rank)
        ensemble_conditional_list.append('interventional')
            
        ensemble_shaps_enet_int = np.zeros_like(shaps_list_enet_int[0])
        for x in shaps_list_enet_int:
            ensemble_shaps_enet_int += x
        ensemble_enet_auc = get_auc(ensemble_shaps_enet_int,beta)
        ensemble_enet_mean_rank = get_mean_ranks(ensemble_shaps_enet_int,beta)
        ensemble_method.append('elastic net')
        ensemble_dataset.append('aml')
        ensemble_noise_list.append(noise)
        ensemble_auc_list.append(ensemble_enet_auc)
        ensemble_mean_rank.append(ensemble_enet_mean_rank)
        ensemble_conditional_list.append('interventional')

        
    AND_dataframe = pd.DataFrame()
    AND_dataframe['method'] = method
    AND_dataframe['dataset'] = dataset
    AND_dataframe['noise'] = noise_list
    AND_dataframe['auc'] = auc_list
    AND_dataframe['mean_rank'] = mean_rank
    AND_dataframe['conditional'] = conditional_list
    AND_dataframe['mse'] = mse_list
    
    ensemble_AND_dataframe = pd.DataFrame()
    ensemble_AND_dataframe['method'] = ensemble_method
    ensemble_AND_dataframe['dataset'] = ensemble_dataset
    ensemble_AND_dataframe['noise'] = ensemble_noise_list
    ensemble_AND_dataframe['auc'] = ensemble_auc_list
    ensemble_AND_dataframe['mean_rank'] = ensemble_mean_rank
    ensemble_AND_dataframe['conditional'] = ensemble_conditional_list
    
    AND_aml_observational_dataframe.to_csv('results/individual_models_for_ensembles_CORR_GROUPS.csv')
    ensemble_AND_aml_observational_dataframe.to_csv('results/ensemble_model_results_CORR_GROUPS.csv')
    
def main():
    ensemble_experiment()

if __name__ == '__main__':
    main()