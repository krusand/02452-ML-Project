from collections import defaultdict
from typing import Iterable

import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, TransformerMixin


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
        raise ValueError("The mode parameter should be one of 'regression' or 'classification'.")

    return error_val


def two_level_cv(k_out: int, 
                 k_in: int, 
                 models: Iterable,
                 X: np.ndarray,
                 y: np.ndarray,
                 mode="regression",
                 seed=131125) -> dict:  
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

    Returns:
    - results:  Dictionary containing the optimal model (hyperparameter value(s)) and the corresponding test error per iteration of the outer loop.
    """
    results = defaultdict(dict)

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

        params = fitted_model.get_params()

        if mode == "regression":
            lam = params["alpha"]
        elif mode == "classification":
            lam = 1 / params["C"]
        
        # append results to results dictionary
        results[i+1]["param_value"] = lam
        results[i+1]["test_error"] = test_error

    return results


class LogTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, columns_to_transform):
        self.columns_to_transform = columns_to_transform
    
    def fit(self, X, y=None):
        self.columns_ = X.columns
        return self
    
    def transform(self, X):
        X = X.copy()
        for col in self.columns_to_transform:
            X[col] = np.log(X[col] + 1/100000)
        return X
    
    def get_feature_names_out(self, *args, **params):
        return self.columns_
