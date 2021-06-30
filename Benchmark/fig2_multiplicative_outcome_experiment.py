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

from datasets import independent_data_multiplicative, corr_group_data_multiplicative, aml_data_multiplicative
from models import MLP, XGBModel

def raw_corrs(X,y,method='pearson'):
    X_frame = pd.DataFrame(X)
    return X_frame.corrwith(pd.Series(y),method=method).abs().values

def linear_coefs_CV(X,y):
    lm = ElasticNetCV(cv=5,l1_ratio=0.99)
    lm.fit(X,y)
    return np.abs(lm.coef_)

def linear_coefs(X,y,alpha=1.0,l1_ratio=0.5):
    # l1_ratio = 0 means L2
    # l1_ratio = 1 means L1
    lm = ElasticNet(alpha=alpha,l1_ratio=l1_ratio)
    lm.fit(X,y)
    return np.abs(lm.coef_)

def rfe_ranks(X,y):
    estimator = SVR(kernel="linear")
    # estimator = ElasticNet(alpha=1,l1_ratio=0)
    selector = RFE(estimator, n_features_to_select=1, step=1)
    selector = selector.fit(X, y)
    return selector.ranking_

def get_mean_ranks(feature_importance,beta):
    return np.mean((101 - rankdata(feature_importance))[np.argwhere(beta == 1)])

def get_mean_ranks_rfe(ranking,beta):
    return np.mean(ranking[np.argwhere(beta == 1)])

def get_auc(feature_importance,beta):
    running_total = [0]
    for x in np.argsort(feature_importance)[::-1]:
        if x in np.where(beta == 1)[0]:
            running_total.append(running_total[-1] + 1)
        else:
            running_total.append(running_total[-1])
    return auc(np.arange(len(running_total)),running_total)

def get_auc_rfe(ranking,beta):
    running_total = [0]
    for x in np.argsort(ranking):
        if x in np.where(beta == 1)[0]:
            running_total.append(running_total[-1] + 1)
        else:
            running_total.append(running_total[-1])
    return auc(np.arange(len(running_total)),running_total)

def independent_data_experiment():
    method = []
    dataset = []
    noise_list = []
    auc_list = []
    mean_rank = []

    for random_seed in range(20):
        print('Iteration {:d}'.format(random_seed))
        noise = 0.0
        X,y,beta = independent_data_multiplicative(noise,random_seed)

        ###
        ### raw corrs

        # fit model
        corrs_feature_attrib = raw_corrs(X,y)

        # calc metrics
        raw_corr_auc = get_auc(corrs_feature_attrib,beta)
        raw_corr_mean_rank = get_mean_ranks(corrs_feature_attrib,beta)

        method.append('pearson')
        dataset.append('independent')
        noise_list.append(noise)
        auc_list.append(raw_corr_auc)
        mean_rank.append(raw_corr_mean_rank)

        ###
        ### linear model
        enet_feature_attrib = linear_coefs_CV(X,y)

        # calc metrics
        enet_auc = get_auc(enet_feature_attrib,beta)
        enet_mean_rank = get_mean_ranks(enet_feature_attrib,beta)

        method.append('elastic net')
        dataset.append('independent')
        noise_list.append(noise)
        auc_list.append(enet_auc)
        mean_rank.append(enet_mean_rank)

        ###
        ### rfe
        output_rankings = rfe_ranks(X,y)
        rfe_auc = get_auc_rfe(output_rankings,beta)
        rfe_mean_rank = get_mean_ranks_rfe(output_rankings,beta)

        method.append('SVMRFE')
        dataset.append('independent')
        noise_list.append(noise)
        auc_list.append(rfe_auc)
        mean_rank.append(rfe_mean_rank)

        ###
        ### xgboost
        xgb_model = XGBModel()
        xgb_model.fit_grid_search(X,y)
        xgb_coefs = xgb_model.get_attribs()

        # calc metrics
        xgb_auc = get_auc(xgb_coefs,beta)
        xgb_mean_rank = get_mean_ranks(xgb_coefs,beta)

        method.append('xgboost')
        dataset.append('independent')
        noise_list.append(noise)
        auc_list.append(xgb_auc)
        mean_rank.append(xgb_mean_rank)

        ###
        ### mlp
        deep_model = MLP()
        deep_model.fit_grid_search(X,y)
        deep_coefs = deep_model.get_attribs()

        # calc metrics
        mlp_auc = get_auc(deep_coefs,beta)
        mlp_mean_rank = get_mean_ranks(deep_coefs,beta)

        method.append('mlp')
        dataset.append('independent')
        noise_list.append(noise)
        auc_list.append(mlp_auc)
        mean_rank.append(mlp_mean_rank)
    
    linear_independent_dataframe = pd.DataFrame()
    linear_independent_dataframe['method'] = method
    linear_independent_dataframe['dataset'] = dataset
    linear_independent_dataframe['noise'] = noise_list
    linear_independent_dataframe['auc'] = auc_list
    linear_independent_dataframe['mean_rank'] = mean_rank
    
    linear_independent_dataframe.to_csv('results/multiplicative_independent_results.csv')

def corr_groups_data_experiment():
    
    method = []
    dataset = []
    noise_list = []
    auc_list = []
    mean_rank = []

    for random_seed in range(20):
        print('Iteration {:d}'.format(random_seed))
        noise = 0
        X,y,beta = corr_group_data_multiplicative(0.99,noise,random_seed)

        ###
        ### raw corrs

        # fit model
        corrs_feature_attrib = raw_corrs(X,y)

        # calc metrics
        raw_corr_auc = get_auc(corrs_feature_attrib,beta)
        raw_corr_mean_rank = get_mean_ranks(corrs_feature_attrib,beta)

        method.append('pearson')
        dataset.append('corr group')
        noise_list.append(noise)
        auc_list.append(raw_corr_auc)
        mean_rank.append(raw_corr_mean_rank)

        ###
        ### linear model
        enet_feature_attrib = linear_coefs_CV(X,y)

        # calc metrics
        enet_auc = get_auc(enet_feature_attrib,beta)
        enet_mean_rank = get_mean_ranks(enet_feature_attrib,beta)

        method.append('elastic net')
        dataset.append('corr group')
        noise_list.append(noise)
        auc_list.append(enet_auc)
        mean_rank.append(enet_mean_rank)

        ###
        ### rfe
        output_rankings = rfe_ranks(X,y)
        rfe_auc = get_auc_rfe(output_rankings,beta)
        rfe_mean_rank = get_mean_ranks_rfe(output_rankings,beta)

        method.append('SVMRFE')
        dataset.append('corr group')
        noise_list.append(noise)
        auc_list.append(rfe_auc)
        mean_rank.append(rfe_mean_rank)

        ###
        ### xgboost
        xgb_model = XGBModel()
        xgb_model.fit_grid_search(X,y)
        xgb_coefs = xgb_model.get_attribs()

        # calc metrics
        xgb_auc = get_auc(xgb_coefs,beta)
        xgb_mean_rank = get_mean_ranks(xgb_coefs,beta)

        method.append('xgboost')
        dataset.append('corr group')
        noise_list.append(noise)
        auc_list.append(xgb_auc)
        mean_rank.append(xgb_mean_rank)

        ###
        ### mlp
        deep_model = MLP()
        deep_model.fit_grid_search(X,y)
        deep_coefs = deep_model.get_attribs()

        # calc metrics
        mlp_auc = get_auc(deep_coefs,beta)
        mlp_mean_rank = get_mean_ranks(deep_coefs,beta)

        method.append('mlp')
        dataset.append('corr group')
        noise_list.append(noise)
        auc_list.append(mlp_auc)
        mean_rank.append(mlp_mean_rank)


    linear_corr_dataframe = pd.DataFrame()
    linear_corr_dataframe['method'] = method
    linear_corr_dataframe['dataset'] = dataset
    linear_corr_dataframe['noise'] = noise_list
    linear_corr_dataframe['auc'] = auc_list
    linear_corr_dataframe['mean_rank'] = mean_rank

    linear_corr_dataframe.to_csv('results/multiplicative_corr_group_results.csv')

def aml_data_experiment():
    method = []
    dataset = []
    noise_list = []
    auc_list = []
    mean_rank = []

    for random_seed in range(20):
        print('Iteration {:d}'.format(random_seed))
        noise = 0
        X,y,beta = aml_data_multiplicative(noise,random_seed)

        ###
        ### raw corrs

        # fit model
        corrs_feature_attrib = raw_corrs(X,y)

        # calc metrics
        raw_corr_auc = get_auc(corrs_feature_attrib,beta)
        raw_corr_mean_rank = get_mean_ranks(corrs_feature_attrib,beta)

        method.append('pearson')
        dataset.append('aml')
        noise_list.append(noise)
        auc_list.append(raw_corr_auc)
        mean_rank.append(raw_corr_mean_rank)

        ###
        ### linear model
        enet_feature_attrib = linear_coefs_CV(X,y)

        # calc metrics
        enet_auc = get_auc(enet_feature_attrib,beta)
        enet_mean_rank = get_mean_ranks(enet_feature_attrib,beta)

        method.append('elastic net')
        dataset.append('aml')
        noise_list.append(noise)
        auc_list.append(enet_auc)
        mean_rank.append(enet_mean_rank)

        ###
        ### rfe
        output_rankings = rfe_ranks(X,y)
        rfe_auc = get_auc_rfe(output_rankings,beta)
        rfe_mean_rank = get_mean_ranks_rfe(output_rankings,beta)

        method.append('SVMRFE')
        dataset.append('aml')
        noise_list.append(1)
        auc_list.append(rfe_auc)
        mean_rank.append(rfe_mean_rank)

        ###
        ### xgboost
        xgb_model = XGBModel()
        xgb_model.fit_grid_search(X,y)
        xgb_coefs = xgb_model.get_attribs()

        # calc metrics
        xgb_auc = get_auc(xgb_coefs,beta)
        xgb_mean_rank = get_mean_ranks(xgb_coefs,beta)

        method.append('xgboost')
        dataset.append('aml')
        noise_list.append(noise)
        auc_list.append(xgb_auc)
        mean_rank.append(xgb_mean_rank)

        ###
        ### mlp
        deep_model = MLP()
        deep_model.fit_grid_search(X,y)
        deep_coefs = deep_model.get_attribs()

        # calc metrics
        mlp_auc = get_auc(deep_coefs,beta)
        mlp_mean_rank = get_mean_ranks(deep_coefs,beta)

        method.append('mlp')
        dataset.append('aml')
        noise_list.append(noise)
        auc_list.append(mlp_auc)
        mean_rank.append(mlp_mean_rank)
        
    linear_aml_dataframe = pd.DataFrame()
    linear_aml_dataframe['method'] = method
    linear_aml_dataframe['dataset'] = dataset
    linear_aml_dataframe['noise'] = noise_list
    linear_aml_dataframe['auc'] = auc_list
    linear_aml_dataframe['mean_rank'] = mean_rank

    linear_aml_dataframe.to_csv('results/multiplicative_aml_results.csv')
    
def main():
    independent_data_experiment()
    corr_groups_data_experiment()
    aml_data_experiment()

if __name__ == '__main__':
    main()
