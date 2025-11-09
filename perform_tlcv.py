import pandas as pd
import sklearn.linear_model as lm

from utils import two_layer_cv, BaselineRegressor, ANNRegressor

models = [BaselineRegressor]

CATEGORICAL_VARIABLES = ["chd", "famhist"]
CONTINUOUS_VARIABLES = ["sbp", "tobacco", "ldl", "typea", "alcohol", "age"]
INCLUDED_VARIABLES = ["sbp", "tobacco", "ldl", "typea", "alcohol", "age", "chd", "famhist"]

df = pd.read_csv("data/heartDisease.csv")
df = df.drop(labels=["row.names"], axis=1)


X, y = df[INCLUDED_VARIABLES], df["obesity"]
print(X)
print(y)


two_layer_cv(k_in=10,
             k_out=10,
             models=models,
             X=X,
             y=y,
             mode="regression")