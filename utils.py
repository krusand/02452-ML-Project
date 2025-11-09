from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
    ClassifierMixin,
    RegressorMixin,
)
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.compose import ColumnTransformer


def compute_error(y_true: np.ndarray,
                  y_pred: np.ndarray,
                  mode="regression") -> float:
    """
    Computes the mean squared error (MSE) for regression models and the error rate for classification models.

    Parameters:
    - y_true:       The true values of the validation or test data. 
    - y_pred:       The values predicted by the model. 
    - mode:         The type of model, regression by default. 

    Returns:
    - error_val:    The computed error.
    """
    if mode == "regression":
        error_val = mean_squared_error(y_true, y_pred)

    elif mode == "classification":
        acc = accuracy_score(y_true, y_pred)
        error_val = 1 - acc

    else:
        raise ValueError(
            "The mode parameter should be one of 'regression' or 'classification'."
        )

    return error_val


def two_layer_cv(k_out: int,
                 k_in: int,
                 models: Iterable,
                 X: np.ndarray,
                 y: np.ndarray,
                 mode="regression",
                 seed=131125,
                 ) -> None:
    """
    Performs two-layer cross-validation for a set of models.

    Parameters:
    - k_out:    Number of folds in the outer loop.
    - k_in:     Number of folds in the inner loop.
    - models:   Iterable containing the models to compare.
    - X:        Features matrix used for training each model.
    - y:        Vector containing the output variable for each data point.
    - mode:     Task performed by the models, either "regression" or "classification".
    - seed:     Seed used for reproducibility purposes.
    """
    results = defaultdict(list)

    # initializing folds for outer loop
    kfold_out = KFold(n_splits=k_out, shuffle=True, random_state=seed)
    
    for i, (train_idx, test_idx) in enumerate(kfold_out.split(X)):
        # split into train and test set
        X_train_out, X_test = X[train_idx], X[test_idx]
        y_train_out, y_test = y[train_idx], y[test_idx]
        val_errors = defaultdict(list)

        # initializing folds for inner loop
        kfold_in = KFold(n_splits=k_in, shuffle=True, random_state=seed)

        for _, (train_idx, val_idx) in enumerate(kfold_in.split(X_train_out)):
            # split into train and validation set
            X_train_in, X_val = X_train_out[train_idx], X_train_out[val_idx]
            y_train_in, y_val = y_train_out[train_idx], y_train_out[val_idx]
            
            for s, model in enumerate(models):
                # fit model on train set
                fitted_model = model.fit(X_train_in, y_train_in)
                
                # get predictions and compute val error
                y_pred = fitted_model.predict(X_val)
                val_error = compute_error(y_val, y_pred, mode=mode) 

                # append results to val_errors dictionary
                model_name = f"model_{s}"
                val_errors[model_name].append(val_error)

        # computing the average validation loss for each model
        averages = {k: np.mean(v) for k, v in val_errors.items()}

        # retrieving the model with the lowest average validation loss
        best_model_name = min(averages, key=averages.get)
        best_model_idx = best_model_name.split("_")[1]
        best_model = models[best_model_idx]

        # train best model on X_train_out and y_train_out
        fitted_model = best_model.fit(X_train_out, y_train_out)

        # get predictions and compute test error
        y_pred = fitted_model.predict(X_test)
        test_error = compute_error(y_test, y_pred, mode=mode)

        # obtain model parameters and name
        params = best_model.get_params()
        model_name = best_model.__class__.__name__

        # identify relevant parameter and its value
        param_dict = {"Ridge": ("lambda", params["alpha"]),
                      "BaselineRegressor": ("-", "-"),
                      "ANNRegressor": ("h", params["hidden_dim"])}
        param_name, param_val = param_dict[model_name]

        # append results to results dictionary 
        results["fold"].append(i+1)
        results["model"].append(model_name)
        results["param_name"].append(param_name)
        results["param_val"].append(param_val)
        results["test_error"].append(test_error)

    # converting results dict to dataframe
    df = pd.DataFrame.from_dict(results)

    # write append results to csv file
    df.to_csv(f'tlcv_results/{mode}_results.csv', mode='a', index=False, header=False)


class LogTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, columns_to_transform):
        self.columns_to_transform = columns_to_transform

    def fit(self, X, y=None):
        self.columns_ = X.columns
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns_to_transform:
            X[col] = np.log(X[col] + 1 / 100000)
        return X

    def get_feature_names_out(self, *args, **params):
        return self.columns_


class Preprocessor():
    def __init__(self, task):
        self.task = task
        assert self.task in {"Classification", "Regression"}, f"Task must equal either 'Classification' or 'Regression', task provided {self.task}"

        if self.task == 'Regression':
            self.COVARIATES = ["sbp", "tobacco", "ldl", "typea", "alcohol", "age", "chd", "famhist"]
            self.INDEPENDENT = ["obesity"]
            self.CATEGORICAL_VARIABLES = ["chd", "famhist"]
            self.CONTINUOUS_VARIABLES = ["sbp", "tobacco", "ldl", "typea", "alcohol", "age"]

        elif self.task == 'Classification':
            self.COVARIATES = ["sbp", "tobacco", "ldl", "typea", "alcohol", "age", "obesity", "famhist"]
            self.INDEPENDENT = ["chd"]
            self.CATEGORICAL_VARIABLES = ["famhist"]
            self.CONTINUOUS_VARIABLES = ["sbp", "tobacco", "ldl", "typea", "alcohol", "age", "obesity"]

        self.num_pipeline = Pipeline(steps=[
        ('log_transform', LogTransformer(["alcohol", "tobacco"])),
        ('scaler', StandardScaler().set_output(transform='pandas'))
        ])

        self.cat_pipeline = Pipeline(steps=[
                ('onehotencoder', OneHotEncoder())
        ])
   
        self.preproc = ColumnTransformer([
            ("num", self.num_pipeline, self.CONTINUOUS_VARIABLES),
            ("cat", self.cat_pipeline, self.CATEGORICAL_VARIABLES),
        ], remainder='passthrough')
        
        return None

    def fit(self, df, y=None):
        if self.task == 'Regression':
            self.preproc.fit(df[self.COVARIATES])
        elif self.task == 'Classification':
            self.preproc.fit(df[self.COVARIATES])
        self.is_fitted_ = True
        self.columns_ = self.preproc.get_feature_names_out()
        return None

    def transform(self, df):
        check_is_fitted(self)
        df_ = df.copy()
        X = df_[self.COVARIATES]
        y = df_[self.INDEPENDENT]

        if self.task == 'Regression':
            X_preprocessed = self.preproc.transform(X)
        elif self.task == 'Classification':
            X_preprocessed = self.preproc.transform(X)
        return pd.DataFrame(X_preprocessed, columns=self.columns_, index=df_.index), y


    def get_feature_names_out(self, *args, **params):
        return self.columns_

class BaselineRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        return None

    def fit(self, X, y):
        # Store training info
        self.n_features_in_ = X.shape[1]
        self.baseline_value = y.mean()
        self.columns_ = X.columns
        self.is_fitted_ = True
        return self

    def predict(self, X):

        check_is_fitted(self)

        return np.repeat(self.baseline_value, X.shape[0])

    def get_feature_names_out(self, *args, **params):
        return self.columns_

    def get_params(self, deep=False):
        return {"Baseline": "None"}


class ANNRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, input_dim, hidden_dim, n_epochs=1000, learning_rate=1e-5):
        self.hidden_dim_ = hidden_dim
        self.input_dim_ = input_dim
        self.learning_rate_ = learning_rate
        self.model_ = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=self.input_dim_, out_features=self.hidden_dim_, bias=True
            ),  # Input layer
            torch.nn.Tanh(),  # Activation function
            torch.nn.Linear(
                in_features=self.hidden_dim_, out_features=1, bias=True
            ),  # Output layer
        )
        self.n_epochs_ = n_epochs
        return None

    def fit(self, X, y, verbose=True):

        self.n_features_in_ = X.shape[1]
        self.columns_ = X.columns

        assert (
            self.n_features_in_ == self.input_dim_
        ), f"Number of columns in data {self.n_features_in_} must equal number of nodes in first layer of ANN {self.input_dim_}"

        self.losses = []
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(
            params=self.model_.parameters(), lr=self.learning_rate_
        )

        for epoch in tqdm(range(self.n_epochs_) if verbose else range(self.n_epochs_)):
            self.model_.train()
            outputs = self.model_(torch.tensor(X.to_numpy()).float()).reshape(-1)
            loss = self.criterion(outputs, torch.tensor(y.to_numpy()).float())

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            self.losses.append(loss.item())

        self.is_fitted_ = True

        return self

    def predict(self, X):
        check_is_fitted(self)
        with torch.no_grad():
            self.model_.eval()
            y_hat = self.model_(torch.tensor(X.to_numpy()).float())
        return y_hat

    def get_feature_names_out(self, *args, **params):
        return self.columns_

    def get_params(self, deep=False):
        return {"hidden_dim": self.hidden_dim_}


class ANNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim, hidden_dim, n_epochs=1000, learning_rate=1e-5):
        self.hidden_dim_ = hidden_dim
        self.input_dim_ = input_dim
        self.learning_rate_ = learning_rate
        self.model_ = torch.nn.Sequential(
                torch.nn.Linear(in_features=self.input_dim_, out_features=self.hidden_dim_, bias=True),     # Input layer
                torch.nn.ReLU(),                                                                # Activation function
                torch.nn.Linear(in_features=self.hidden_dim_, out_features=2, bias=True),    # Output layer
                torch.nn.Sigmoid()
        )
        self.n_epochs_ = n_epochs
        return None

    def fit(self, X, y, verbose=True):

        self.n_features_in_ = X.shape[1]
        self.columns_ = X.columns

        assert (
            self.n_features_in_ == self.input_dim_
        ), f"Number of columns in data {self.n_features_in_} must equal number of nodes in first layer of ANN {self.input_dim_}"

        self.losses = []
        self.criterion = torch.nn.BCELoss()
        self.optimizer = torch.optim.SGD(
            params=self.model_.parameters(), lr=self.learning_rate_
        )

        for epoch in tqdm(range(self.n_epochs_) if verbose else range(self.n_epochs_)):
            self.model_.train()
            outputs = self.model_(torch.tensor(X.to_numpy()).float()).reshape(-1)
            loss = self.criterion(outputs, torch.tensor(y.to_numpy()).float())

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            self.losses.append(loss.item())

        self.is_fitted_ = True

        return self

    def predict(self, X):
        check_is_fitted(self)
        with torch.no_grad():
            self.model_.eval()
            y_hat = self.model_(torch.tensor(X.to_numpy()).float())
        return y_hat

    def get_feature_names_out(self, *args, **params):
        return self.columns_

    def get_params(self, deep=False):
        return {"hidden_dim": self.hidden_dim_}


