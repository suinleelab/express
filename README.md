# EXPRESS
Code for paper "Uncovering expression signatures of synergistic drug response in acute myeloid leukemia by explaining AI ensembles"

The "Benchmark" directory contains the scripts necessary to run the feature discovery benchmark experiments corresponding to Fig. 2 and Fig. 3 in the paper.

The "AMLAnalysis" directory contains the scripts necessary to run the AML data analysis experiments corresponding to Fig. 4, Fig. 5, and Fig. 6 in the paper.

This software was originally designed and run on a system running Ubuntu 16.04.3 with Python 3.3.6. For neural network model training and interpretation, we used a single Nvidia GeForce GTX 980 Ti GPU, though we anticipate that other GPUs will also work. Standard python software packages used: PyTorch (1.9.0), XGBoost (1.4.2), scikit-learn (0.24.2), numpy (1.21.2), scipy (1.7.1), pandas (1.3.2), matplotlib (3.4.3), seaborn (0.11.2), networkx (2.6.2), tqdm (4.62.1). For model interpretability, we additionally used the following Python software packages available here: SHAP (0.39.0), and SAGE (0.0.4). To interface to the R language through Python, we used the rpy2 (3.4.5) library, which requires an existing R (4.1.1) installation. The following modules from the Python Standard Library were also used: pickle, random, copy, itertools.
