import pandas as pd
import sklearn.linear_model as lm
import numpy as np

from utils import two_layer_cv, BaselineRegressor, ANNRegressor, Preprocessor, BaselineClassifier, ANNClassifier

# hyperparameter values to test
hds = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
lams = np.logspace(np.log10(1e-5), np.log10(1e3), num=200)


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

# preprocessing data for regression
PreProp_reg = Preprocessor(task="regression", covariates=["sbp","tobacco","ldl","typea","alcohol","age","chd","famhist"])
PreProp_reg.fit(df)
X_reg, y_reg = PreProp_reg.transform(df)

# performing two-level cross-validation for the regression models
for models in models_list_reg:
    two_layer_cv(k_in=10,
                 k_out=10,
                 models=models,
                 X=X_reg,
                 y=y_reg,
                 mode="regression")

# preprocessing data for classification
PreProp_clf = Preprocessor(task="classification")
PreProp_clf.fit(df)
X_clf, y_clf = PreProp_clf.transform(df)

# performing two-level cross-validation for the classification models
for models in models_list_clf:
    two_layer_cv(k_in=10,
                 k_out=10,
                 models=models,
                 X=X_clf,
                 y=y_clf,
                 mode="classification")
