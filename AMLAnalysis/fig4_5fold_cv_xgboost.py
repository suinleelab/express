# nested 5-fold CV procedure with xgboost models used as an example
# folds are split randomly (i.e., not accounting for patients, combos, drugs)
# test fold is only required to contain _samples_ not seen in training data
# this corresponds to the top plot in Fig. 4
# in subsequent plots, test fold is required to contain 
#    (combinations of drugs/ patients / drugs) not seen in the training data

# import libraries
import pandas as pd
import seaborn as sb
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import ShuffleSplit
import random
import xgboost
from tqdm import *

# list of parameter settings to search
params1 = {'max_depth':4,
          "tree_method":'approx',
          'subsample':0.8,
          'learning_rate':0.05,
          'n_estimators':500}
params2 = {'max_depth':6,
          "tree_method":'approx',
          'subsample':0.8,
          'learning_rate':0.05,
          'n_estimators':500}
params3 = {'max_depth':8,
          "tree_method":'approx',
          'subsample':0.8,
          'learning_rate':0.05,
          'n_estimators':500}
params4 = {'max_depth':4,
          "tree_method":'approx',
          'subsample':1,
          'learning_rate':0.05,
          'n_estimators':500}
params5 = {'max_depth':6,
          "tree_method":'approx',
          'subsample':1,
          'learning_rate':0.05,
          'n_estimators':500}
params6 = {'max_depth':8,
          "tree_method":'approx',
          'subsample':1,
          'learning_rate':0.05,
          'n_estimators':500}
params7 = {'max_depth':4,
          "tree_method":'approx',
          'subsample':0.8,
          'learning_rate':0.1,
          'n_estimators':500}
params8 = {'max_depth':6,
          "tree_method":'approx',
          'subsample':0.8,
          'learning_rate':0.1,
          'n_estimators':500}
params9 = {'max_depth':8,
          "tree_method":'approx',
          'subsample':0.8,
          'learning_rate':0.1,
          'n_estimators':500}
params10 = {'max_depth':4,
          "tree_method":'approx',
          'subsample':1,
          'learning_rate':0.1,
          'n_estimators':500}
params11 = {'max_depth':6,
          "tree_method":'approx',
          'subsample':1,
          'learning_rate':0.1,
          'n_estimators':500}
params12 = {'max_depth':8,
          "tree_method":'approx',
          'subsample':1,
          'learning_rate':0.1,
          'n_estimators':500}

params_list = [params1,params2,params3,params4,params5,params6,params7,params8,params9,params10,params11,params12]

def norm(X_df, mean=None, std=None, zero_cols=None):
    
    X = X_df.values
    if std is None:
        std = np.nanstd(X, axis=0)
    if zero_cols is None:
        zero_cols = std!=0
    X = X[:,zero_cols]
    X = np.ascontiguousarray(X)
    if mean is None:
        mean = np.mean(X, axis=0)
    X = (X-mean)/std[zero_cols]
    new_cols = X_df.columns[zero_cols]
    X_df = pd.DataFrame(data=X,columns = new_cols, index = X_df.index)
    return(X_df, mean, std, zero_cols)



def fiveFoldCV():

	# load and concatenate design matrix
	X_drug_labels = pickle.load(open('../data/X_drug_labels_train.p','rb'))
	X_drug_targets = pickle.load(open('../data/X_drug_targets_train.p','rb'))
	X_rna_seq_full = pickle.load(open('../data/X_rna_seq_full_train.p','rb'))

	X = pd.concat([X_drug_labels,X_drug_targets,X_rna_seq_full],axis = 1)
	y = pickle.load(open('../data/y.p','rb'))

	# initialize lists to hold best test errors and hyperparameters
	rand_xgb_mse_list = []
	rand_xgb_best_hps_list = []
	rand_xgb_best_val_mses = []

	# 5-fold cv
	n_estimators=500
	ts = ShuffleSplit(n_splits=5, test_size=.2, random_state=1017)
	ts.get_n_splits(X)
	fold=0
	# outer loop of 5 test folds
	for train_index, test_index in ts.split(X):
	
		best_hps = params_list[0]
		best_mse = np.inf
	
		X_train = X.iloc[train_index,:]
		X_test = X.iloc[test_index,:]
		y_train = y[train_index]
		y_test = y[test_index]
	
		vs = ShuffleSplit(n_splits=1, test_size=.2, random_state=1017)
		vs.get_n_splits(X_train)
	
		for tr_index, val_index in vs.split(X_train):
			X_tr = X_train.iloc[tr_index,:]
			X_val = X_train.iloc[val_index,:]
			y_tr = y_train[tr_index]
			y_val = y_train[val_index]
	
		for i in range(12):
		
			print('outer loop (test fold) = {}, inner loop (/12 parameter settings) = {}'.format(fold,i))
			params = params_list[i]
		
			params['base_score'] = np.mean(y_tr)
			params['verbose'] = False
		
			# normalize the parameter tuning data

			X_tr, mean, std, zero_cols = norm(X_tr)
			X_val, mean, std, zero_cols = norm(X_val, mean, std, zero_cols=zero_cols)
		
			### FIT MODEL AND PREDICT
			results = {}
			dtrain = xgboost.DMatrix(X_tr,label=y_tr)
			dtest = xgboost.DMatrix(X_val,label=y_val)
			# Train XGBoost
			bst = xgboost.train(params,dtrain,n_estimators,[(dtrain,"train")],evals_result=results)
			y_preds = bst.predict(dtest)
		
			mse = mean_squared_error(y_val, y_preds)
		
			if mse<best_mse:
				print("Fold {}, cv mse {}".format(fold,mse))
				print(params)
				best_hps = params
				best_mse = mse
	
		rand_xgb_best_hps_list.append(best_hps)
		rand_xgb_best_val_mses.append(best_mse)
	
		# re-set best params
	
		params = best_hps
		
		params['base_score'] = np.mean(y_train)
		params['verbose'] = False
	
		# normalize testing data

		X_train, mean, std, zero_cols = norm(X_train)
		X_test, mean, std, zero_cols = norm(X_test, mean, std, zero_cols=zero_cols)
	
		## test model
		results = {}
		
		dtrain = xgboost.DMatrix(X_train,label=y_train)
		dtest = xgboost.DMatrix(X_test,label=y_test)

		# Train XGBoost
		bst = xgboost.train(params,dtrain,n_estimators,[(dtrain,"train")],evals_result=results)
		preds = bst.predict(dtest)
		rand_xgb_preds[test_index] = preds
	
		y_true = [x > np.std(y_test) for x in y_test]
	
		rand_xgb_mse_list.append(mean_squared_error(y_test,preds))
	
		print('test fold = {}'.format(fold))
		print('mse = {}'.format(mean_squared_error(y_test,preds)))
	
		fold+=1
		
	pickle.dump((rand_xgb_mse_list,rand_xgb_best_hps_list,rand_xgb_best_val_mses),
           open('results/rand_xgb_metrics.p','wb'))

def main():
    fiveFoldCV()

if __name__ == '__main__':
    main()
