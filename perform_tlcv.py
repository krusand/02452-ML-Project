import pandas as pd
import sklearn.linear_model as lm

from utils import two_layer_cv, BaselineRegressor, ANNRegressor, Preprocessor

models = [BaselineRegressor]


df = pd.read_csv("data/heartDisease.csv")

PreProp = Preprocessor(task="Regression")
PreProp.fit(df)
X, y = PreProp.transform(df)


two_layer_cv(k_in=10,
             k_out=10,
             models=models,
             X=X,
             y=y,
             mode="regression")