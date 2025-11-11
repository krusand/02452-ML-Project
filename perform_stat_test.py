import pandas as pd
import sklearn.linear_model as lm
import numpy as np
from utils import BaselineRegressor, ANNRegressor, Preprocessor, BaselineClassifier, ANNClassifier, performance_diff_test

FEATURES_REGRESSION = ["sbp","tobacco","ldl","typea","alcohol","age","chd","famhist"]
OUTCOME_REGRESSION = ["obesity"]
FEATURES_CLASSIFICATION = [ "sbp", "tobacco", "ldl", "typea", "alcohol", "age", "obesity", "famhist"]
OUTCOME_CLASSIFICATION = ["chd"]

regression_results = pd.read_csv("tlcv_results/regression_results.csv")
classification_results = pd.read_csv("tlcv_results/classification_results.csv")
regression_results_grouped = regression_results.groupby(["model", "param_name"], dropna=False)["param_val"].agg(pd.Series.mode).to_frame().reset_index()
classification_results_grouped = classification_results.groupby(["model", "param_name"], dropna=False)["param_val"].agg(pd.Series.mode).to_frame().reset_index()

reg_ridge_lam_parameter = regression_results_grouped[(regression_results_grouped["model"] == "Ridge") & (regression_results_grouped["param_name"] == "lambda")]["param_val"].values[0]
if np.array(reg_ridge_lam_parameter).shape != ():
    reg_ridge_lam_parameter = reg_ridge_lam_parameter[0]
reg_ridge_lam_parameter = float(reg_ridge_lam_parameter)

reg_ann_h_parameter = int(regression_results_grouped[(regression_results_grouped["model"] == "ANNRegressor") & (regression_results_grouped["param_name"] == "h")]["param_val"].values[0])
if np.array(reg_ann_h_parameter).shape != ():
    reg_ann_h_parameter = int(reg_ann_h_parameter[0])

clf_logistic_lam_parameter = classification_results_grouped[(classification_results_grouped["model"] == "LogisticRegression") & (classification_results_grouped["param_name"] == "lambda")]["param_val"].values[0]
if np.array(clf_logistic_lam_parameter).shape != ():
    clf_logistic_lam_parameter = clf_logistic_lam_parameter[0]
clf_logistic_lam_parameter = float(clf_logistic_lam_parameter)

clf_ann_h_parameter = int(classification_results_grouped[(classification_results_grouped["model"] == "ANNClassifier") & (classification_results_grouped["param_name"] == "h")]["param_val"].values[0])
if np.array(clf_ann_h_parameter).shape != ():
    clf_ann_h_parameter = int(clf_ann_h_parameter[0])




clf_cv_seed_used = int(classification_results["cv_seed_used"].drop_duplicates().values[0])
reg_cv_seed_used = int(regression_results["cv_seed_used"].drop_duplicates().values[0])



# regression models
base_reg = BaselineRegressor()
lin_reg = lm.Ridge(alpha=reg_ridge_lam_parameter)
ann_reg = ANNRegressor(hidden_dim=reg_ann_h_parameter)

# classification models
base_clf = BaselineClassifier()
log_reg_clf = lm.LogisticRegression(penalty="l2", C=1/clf_logistic_lam_parameter)
ann_clf = ANNClassifier(hidden_dim=clf_ann_h_parameter)

# loading the data
df = pd.read_csv("data/heartDisease.csv")

# preprocessing data for regression
PreProp_reg = Preprocessor(task="regression", covariates=FEATURES_REGRESSION, independent=OUTCOME_REGRESSION)
PreProp_reg.fit(df)
X_reg, y_reg = PreProp_reg.transform(df)

# regression model comparisons
model_comparisons_reg = [(lin_reg, base_reg)]

for m1, m2 in model_comparisons_reg:
    p_val, lower, upper = performance_diff_test(mode="regression", model_1=m1, model_2=m2, X=X_reg, y=y_reg, seed=reg_cv_seed_used)
print(f"p_val: {p_val}")
print(f"CI: [{lower:.3f}, {upper:.3f}]")
# preprocessing data for classification
PreProp_clf = Preprocessor(task="classification", covariates=FEATURES_CLASSIFICATION, independent=OUTCOME_CLASSIFICATION)
PreProp_clf.fit(df)
X_clf, y_clf = PreProp_clf.transform(df)

# classification model comparisons
model_comparisons_clf = [(log_reg_clf, ann_clf)]

for m1, m2 in model_comparisons_clf:
    p_val, lower, upper = performance_diff_test(mode="classification", model_1=m1, model_2=m2, X=X_clf, y=y_clf, seed=clf_cv_seed_used)