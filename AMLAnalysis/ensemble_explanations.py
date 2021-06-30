##
## ensemble 100 models for explanations
##

# import dependencies
import pandas as pd
import seaborn as sb
import numpy as np
import pickle

import random
import xgboost

import shap

def ensembleExplanations():

	# load and concatenate design matrix
	X_drug_labels = pickle.load(open('../data/X_drug_labels_train.p','rb'))
	X_drug_targets = pickle.load(open('../data/X_drug_targets_train.p','rb'))
	X_rna_seq_full = pickle.load(open('../data/X_rna_seq_full_train.p','rb'))

	X = pd.concat([X_drug_labels,X_drug_targets,X_rna_seq_full],axis = 1)
	y = pickle.load(open('../data/y.p','rb'))

	individual_explanations = []
	nmodels = 100

	for i in range(nmodels):
	
		dtrain = xgboost.DMatrix(X, label=y)

		params = {'max_depth':6,
				  "tree_method":'approx',
				  'subsample':0.8,
				  'colsample_bytree':0.5,
				  'learning_rate':0.05,
				  'base_score':np.mean(y),
				  'n_estimators':500,
				  'seed':i}
		nrounds=500
	
		results = {}

		# Train XGBoost
		bst = xgboost.train(params,dtrain,nrounds,[(dtrain,"train")],evals_result=results)
	
		shaps = shap.TreeExplainer(bst).shap_values(X)
	
		individual_explanations.append(shaps)

	ensembled_explanations = np.zeros(individual_explanations[0].shape)

	for i,sh in enumerate(individual_explanations):
		ensembled_explanations += sh

	ensembled_explanations /= 100.

	pickle.dump(ensembled_explanations,open('results/ensembled_explanations.p','wb'))

def main():
    ensembleExplanations()

if __name__ == '__main__':
    main()