import pandas as pd
import sklearn.linear_model as lm
import numpy as np

from utils import two_layer_cv, BaselineRegressor, ANNRegressor, Preprocessor, BaselineClassifier, ANNClassifier

# hyperparameter values to test
hds = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
lams = np.logspace(np.log10(1e-5), np.log10(1e3), num=200)

FEATURES_REGRESSION = ["sbp","tobacco","ldl","typea","alcohol","age","chd","famhist"]
OUTCOME_REGRESSION = ["obesity"]
FEATURES_CLASSIFICATION = [ "sbp", "tobacco", "ldl", "typea", "alcohol", "age", "obesity", "famhist"]
OUTCOME_CLASSIFICATION = ["chd"]

# regression models to compare
models_list_reg = [[ANNRegressor(input_dim=8,hidden_dim=hd, verbose=False) for hd in hds],
                   [BaselineRegressor()], 
                   [lm.Ridge(alpha=a) for a in lams],
                   ]

# classification models to compare
models_list_clf = [[ANNClassifier(input_dim=8,hidden_dim=hd, verbose=False) for hd in hds],
                   [BaselineClassifier()], 
                   [lm.LogisticRegression(penalty="l2", C=1/lam) for lam in lams],
                   ]

# loading the data
df = pd.read_csv("data/heartDisease.csv")

# performing two-level cross-validation for the regression models
for models in models_list_reg:
    two_layer_cv(k_in=10,
                 k_out=10,
                 models=models,
                 df=df.copy(),
                 mode="regression",
                 features=FEATURES_REGRESSION,
                 outcome=OUTCOME_REGRESSION)

# performing two-level cross-validation for the classification models
for models in models_list_clf:
    two_layer_cv(k_in=10,
                 k_out=10,
                 models=models,
                 df=df.copy(),
                 mode="classification",
                 features=FEATURES_CLASSIFICATION,
                 outcome=OUTCOME_CLASSIFICATION)
    
