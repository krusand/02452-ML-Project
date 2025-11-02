from collections import defaultdict
from typing import Iterable

import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split


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


def two_layer_cv(k1: int, 
                 k2: int, 
                 models: Iterable,
                 X: np.ndarray,
                 y: np.ndarray,
                 mode="regression",
                 seed=131125) -> dict:  
    """
    Performs two-layer cross-validation for a set of models. 

    Parameters:
    - k1:       Number of folds in the outer loop.
    - k2:       Number of folds in the inner loop.
    - models:   Iterable containing the models to compare.
    - X:        Features matrix used for training each model.
    - y:        Vector containing the output variable for each data point. 
    - mode:     Task performed by the models, either "regression" or "classification".
    - seed:     Seed used for reproducibility purposes. 

    Returns:
    - results:  Dictionary containing the optimal model (hyperparameter value(s)) and the corresponding test error per iteration of the outer loop.
    """
    results = defaultdict(dict)

    for i in range(k1):
        val_errors = defaultdict(list)
        X_train_outer, X_test_outer, y_train_outer, y_test_outer = train_test_split(X, 
                                                                                    y, 
                                                                                    test_size=1/k1, 
                                                                                    random_state=seed)

        for _ in range(k2):
            X_train_inner, X_test_inner, y_train_inner, y_test_inner = train_test_split(X_train_outer, 
                                                                                        y_train_outer, 
                                                                                        test_size=1/k2, 
                                                                                        random_state=seed)
            
            for s, model in enumerate(models):
                fitted_model = model.fit(X_train_inner, y_train_inner)
                y_pred = fitted_model.predict(X_test_inner)

                error_val = compute_error(y_test_inner, y_pred, mode=mode) 
                model_name = f"model_{s}"
                val_errors[model_name].append(error_val)

        params = fitted_model.get_params()

        if mode == "regression":
            lam = params["alpha"]
        elif mode == "classification":
            lam = 1 / params["C"]
        
    return results