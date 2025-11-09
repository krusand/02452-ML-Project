import pandas as pd
import sklearn.linear_model as lm

from utils import two_layer_cv, BaselineRegressor, ANNRegressor, Preprocessor

models_list = [[BaselineRegressor()], 
               [lm.Ridge(alpha=14)],
               [lm.Ridge(alpha=15)],
               ]

df = pd.read_csv("data/heartDisease.csv")

PreProp = Preprocessor(task="regression")
PreProp.fit(df)
X, y = PreProp.transform(df)

for models in models_list:
    two_layer_cv(k_in=10,
                k_out=10,
                models=models,
                X=X,
                y=y,
                mode="regression")