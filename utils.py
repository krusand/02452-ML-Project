from collections import defaultdict
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from scipy.stats import beta, binom, t
import torch
from tqdm import tqdm

from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import KFold

from sklearn.utils.validation import check_is_fitted

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
    ClassifierMixin,
    RegressorMixin,
)
from sklearn.compose import ColumnTransformer

BASE_CONT_VARIABLES = ["sbp", "tobacco", "ldl", "typea", "alcohol", "age", "adiposity", "obesity"]
BASE_CAT_VARIABLES = ["chd", "famhist"]


def compute_error(y_true: np.ndarray, y_pred: np.ndarray, mode="regression") -> float:
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


def two_layer_cv(
    k_out: int,
    k_in: int,
    models: Iterable,
    df: pd.DataFrame,
    mode="regression",
    seed=131125,
    features: list[str] | None = None,
    outcome: list[str] | None = None
) -> None:
    """
    Performs two-layer cross-validation for a set of models.

    Parameters:
    - k_out:    Number of folds in the outer loop.
    - k_in:     Number of folds in the inner loop.
    - models:   Iterable containing the models to compare.
    - df:       Features and outcome
    - mode:     Task performed by the models, either "regression" or "classification".
    - seed:     Seed used for reproducibility purposes.
    """
    results = defaultdict(list)

    # initializing folds for outer loop
    kfold_out = KFold(n_splits=k_out, shuffle=True, random_state=seed)

    for i, (train_idx, test_idx) in enumerate(tqdm(kfold_out.split(df), total=k_out)):
        df_train_out = df.iloc[train_idx]
        df_test_out = df.iloc[test_idx]

        val_errors = defaultdict(list)

        # initializing folds for inner loop
        kfold_in = KFold(n_splits=k_in, shuffle=True, random_state=seed)

        for _, (train_idx, val_idx) in enumerate(kfold_in.split(df_train_out)):
            df_train_in = df_train_out.iloc[train_idx]
            df_val = df_train_out.iloc[val_idx]

            # fit pipeline only on training
            preprocessor_in = Preprocessor(task=mode, covariates=features, independent=outcome)
            X_train_in, y_train_in = preprocessor_in.fit_transform(df_train_in)
            X_val, y_val = preprocessor_in.transform(df_val)

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
        best_model_idx = int(best_model_name.split("_")[1])
        best_model = models[best_model_idx]
        
        # fit pipeline only on training
        preprocessor_out = Preprocessor(task=mode, covariates=features, independent=outcome)
        X_train_out, y_train_out = preprocessor_out.fit_transform(df_train_out)
        X_test, y_test = preprocessor_out.transform(df_test_out)

        # train best model on X_train_out and y_train_out
        fitted_model = best_model.fit(X_train_out, y_train_out)

        # get predictions and compute test error
        y_pred = fitted_model.predict(X_test)
        test_error = compute_error(y_test, y_pred, mode=mode)

        # obtain model parameters and name
        params = best_model.get_params()
        model_name = best_model.__class__.__name__

        if model_name == "Ridge":
            param_name, param_val = "lambda", params["alpha"]
        elif model_name == "BaselineRegressor" or model_name == "BaselineClassifier":
            param_name, param_val = None, None
        elif model_name == "ANNRegressor" or model_name == "ANNClassifier":
            param_name, param_val = "h", params["hidden_dim"]
        elif model_name == "LogisticRegression":
            param_name, param_val = "lambda", 1 / params["C"]

        # append results to results dictionary
        results["fold"].append(i + 1)
        results["model"].append(model_name)
        results["param_name"].append(param_name)
        results["param_val"].append(param_val)
        results["test_error"].append(test_error)
        results["cv_seed_used"].append(seed)

    # converting results dict to dataframe
    df = pd.DataFrame.from_dict(results)

    # write append results to csv file
    df.to_csv(f"tlcv_results/{mode}_results.csv", mode="a", index=False, header=False)


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


class ModifiedLabelEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, columns_to_transform):
        self.columns_to_transform = columns_to_transform
        self.LabelEncoderDict = {
            f"{col}": LabelEncoder() for col in self.columns_to_transform
        }

    def fit(self, X, y=None):
        self.columns_ = X.columns
        for col in self.columns_to_transform:
            self.LabelEncoderDict[col].fit(X[col])
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns_to_transform:
            X[col] = self.LabelEncoderDict[col].transform(X[col])
        return X

    def fit_transform(self, X, y=None, **fit_params):
        return super().fit_transform(X, y, **fit_params)

    def get_feature_names_out(self, *args, **params):
        return self.columns_


class Preprocessor:
    def __init__(self, task, covariates=None, independent=None):
        self.task = task
        assert self.task in {
            "classification",
            "regression",
        }, f"Task must equal either 'classification' or 'regression', task provided {self.task}"

        if self.task == "regression":
            self.COVARIATES = (
                covariates
                if covariates is not None
                else [
                    "sbp",
                    "tobacco",
                    "ldl",
                    "typea",
                    "alcohol",
                    "age",
                    "chd",
                    "famhist",
                    "adiposity",
                ]
            )
            self.INDEPENDENT = independent if independent is not None else ["obesity"]
            self.CATEGORICAL_VARIABLES = list(
                set(BASE_CAT_VARIABLES).intersection(self.COVARIATES)
            )
            self.CONTINUOUS_VARIABLES = list(
                set(BASE_CONT_VARIABLES).intersection(self.COVARIATES)
            )

        elif self.task == "classification":
            self.COVARIATES = (
                covariates
                if covariates is not None
                else [
                    "sbp",
                    "tobacco",
                    "ldl",
                    "typea",
                    "alcohol",
                    "age",
                    "obesity",
                    "famhist",
                ]
            )
            self.INDEPENDENT = independent if independent is not None else ["chd"]
            self.CATEGORICAL_VARIABLES = list(
                set(BASE_CAT_VARIABLES).intersection(self.COVARIATES)
            )
            self.CONTINUOUS_VARIABLES = list(
                set(BASE_CONT_VARIABLES).intersection(self.COVARIATES)
            )

        self.num_pipeline = Pipeline(
            steps=[
                ("log_transform", LogTransformer(list(set(["alcohol", "tobacco"]).intersection(self.CONTINUOUS_VARIABLES)))),
                ("scaler", StandardScaler().set_output(transform="pandas")),
            ]
        )

        self.cat_pipeline = Pipeline(
            steps=[("labelencoder", ModifiedLabelEncoder(self.CATEGORICAL_VARIABLES))]
        )

        self.preproc = ColumnTransformer(
            [
                ("num", self.num_pipeline, self.CONTINUOUS_VARIABLES),
                ("cat", self.cat_pipeline, self.CATEGORICAL_VARIABLES),
            ],
            remainder="passthrough",
        )

        return None

    def fit(self, df):
        if self.task == "regression":
            self.preproc.fit(df[self.COVARIATES])
        elif self.task == "classification":
            self.preproc.fit(df[self.COVARIATES])
        self.is_fitted_ = True
        self.columns_ = self.preproc.get_feature_names_out()
        return None

    def transform(self, df):
        check_is_fitted(self)
        df_ = df.copy()
        X = df_[self.COVARIATES]
        y = df_[self.INDEPENDENT]

        if self.task == "regression":
            X_preprocessed = self.preproc.transform(X)
        elif self.task == "classification":
            X_preprocessed = self.preproc.transform(X)
        return pd.DataFrame(X_preprocessed, columns=self.columns_, index=df_.index), y

    def fit_transform(self, df):
        self.fit(df)
        X, y = self.transform(df)
        return X, y

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

        return np.array(np.repeat(self.baseline_value, X.shape[0]))

    def get_feature_names_out(self, *args, **params):
        return self.columns_

    def get_params(self, deep=False):
        return {"Baseline": "None"}


class BaselineClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        return None

    def fit(self, X, y):
        # Store training info
        self.n_features_in_ = X.shape[1]
        self.baseline_value = y.mode()
        self.columns_ = X.columns
        self.is_fitted_ = True
        return self

    def predict(self, X):

        check_is_fitted(self)

        return np.array(np.repeat(self.baseline_value, X.shape[0]))

    def get_feature_names_out(self, *args, **params):
        return self.columns_

    def get_params(self, deep=False):
        return {"Baseline": "None"}


class ANNRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self, hidden_dim, input_dim=8, n_epochs=2500, learning_rate=1e-5, verbose=True
    ):
        self.hidden_dim_ = hidden_dim
        self.input_dim_ = input_dim
        self.learning_rate_ = learning_rate
        self.verbose = verbose
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

    def fit(self, X, y):

        self.n_features_in_ = X.shape[1]
        self.columns_ = X.columns

        assert (
            self.n_features_in_ == self.input_dim_
        ), f"Number of columns in data {self.n_features_in_} must equal number of nodes in first layer of ANN {self.input_dim_}"

        self.losses = []
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            params=self.model_.parameters(), lr=self.learning_rate_
        )

        for epoch in range(self.n_epochs_):
            self.model_.train()
            outputs = self.model_(torch.tensor(X.to_numpy()).float()).reshape(-1)
            loss = self.criterion(
                outputs, torch.tensor(y.to_numpy()).float().reshape(-1)
            )

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
        return y_hat.detach().numpy().reshape(-1)

    def get_feature_names_out(self, *args, **params):
        return self.columns_

    def get_params(self, deep=False):
        return {"hidden_dim": self.hidden_dim_}


class ANNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self, hidden_dim, input_dim=8, n_epochs=2500, learning_rate=1e-5, verbose=True
    ):
        self.hidden_dim_ = hidden_dim
        self.input_dim_ = input_dim
        self.learning_rate_ = learning_rate
        self.model_ = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=self.input_dim_, out_features=self.hidden_dim_, bias=True
            ),  # Input layer
            torch.nn.ReLU(),  # Activation function
            torch.nn.Linear(
                in_features=self.hidden_dim_, out_features=2, bias=True
            ),  # Output layer
        )
        self.n_epochs_ = n_epochs
        return None

    def fit(self, X, y):

        self.n_features_in_ = X.shape[1]
        self.columns_ = X.columns

        assert (
            self.n_features_in_ == self.input_dim_
        ), f"Number of columns in data {self.n_features_in_} must equal number of nodes in first layer of ANN {self.input_dim_}"

        self.losses = []
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            params=self.model_.parameters(), lr=self.learning_rate_
        )

        for epoch in range(self.n_epochs_):
            self.model_.train()
            outputs = self.model_(torch.tensor(X.to_numpy()).float())
            loss = self.criterion(outputs, torch.tensor(y.to_numpy()).long().squeeze())

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
            logits = self.model_(torch.tensor(X.to_numpy()).float())
            probs = torch.softmax(logits, dim=1)
            y_hat = np.argmax(probs, axis=1).unsqueeze(1)
        return y_hat.detach().numpy().reshape(-1)

    def get_feature_names_out(self, *args, **params):
        return self.columns_

    def get_params(self, deep=False):
        return {"hidden_dim": self.hidden_dim_}



def performance_diff_test(
    mode, model_1, model_2, X, y, K=10, conf_level=0.95, seed=131125
) -> Tuple[float, float, float]:
    """
    When mode=regression: Performs a paired t-test to assess whether the difference in performance between two regression models is significant.
    When mode=classification: Performs McNemar's test to assess whether the difference in performance between two classification models is significant.

    Parameters:
    - mode: Specifies the model types, either regression or classification.
    - model_1: The first model to be compared.
    - model_2: The second model to be compared.
    - X: Features to be used as input to the models.
    - y: Labels to be used as input to the models.
    - K: K used for K-fold cross-validation.
    - conf_level: Confidence level for confidence interval.

    Returns:
    - p_val: The p-value associated with the null hypothesis stating that there is no difference in performance between model_1 and model_2.
    - lower: The lower b ound of the confidence interval.
    - upper: The upper bound of the confidence interval.
    """
    assert mode in {
        "classification",
        "regression",
    }, f"mode must equal either 'classification' or 'regression', mode provided {mode}"

    # dictionary to contain results per fold
    fold_results = defaultdict(list)

    # initializing number of test objects
    n = 0

    # initializing folds for outer loop
    kfold_out = KFold(n_splits=K, shuffle=True, random_state=seed)

    for i, (train_idx, test_idx) in enumerate(kfold_out.split(X)):
        # define train and test set for fold
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx].values.ravel()

        # adding to the value of test objects
        n += len(y_test)

        # fit models   

        fitted_model_1 = model_1.fit(X_train, y_train)
        fitted_model_2 = model_2.fit(X_train, y_train)

        # get predictions
        pred_1 = fitted_model_1.predict(X_test)
        pred_2 = fitted_model_2.predict(X_test)

        # compute difference in squared error for regression or agreements/disagreements for classification
        if mode == "regression":
            # computing sum of difference in squared losses
            z_1 = (pred_1 - y_test)**2
            z_2 = (pred_2 - y_test)**2
            z_i = z_1 - z_2

            # adding result to fold_results
            for z_val in z_i:
                fold_results["z"].append(z_val)
        
        elif mode == "classification":
            # computing agreements and disagreements
            model_1_binary = pred_1 == y_test
            model_2_binary = pred_2 == y_test

            # computing agreements/disagreements
            n_11 = np.sum((model_1_binary == 1) & (model_2_binary == 1))
            n_12 = np.sum((model_1_binary == 1) & (model_2_binary == 0))
            n_21 = np.sum((model_1_binary == 0) & (model_2_binary == 1))
            n_22 = np.sum((model_1_binary == 0) & (model_2_binary == 0))
            
            # adding counts to fold_results
            fold_results["n_11"].append(n_11)
            fold_results["n_12"].append(n_12)
            fold_results["n_21"].append(n_21)
            fold_results["n_22"].append(n_22)

        # adding fold number to fold_results
        fold_results["fold"].append(i)   # delete if not using this key

    # compute p-value and confidence interval bounds
    alpha = 1 - conf_level

    if mode == "regression":
        # vector containing all z_i values
        z = fold_results["z"]

        # average z_i value (eq 11.52c in the ML book)
        z_hat = np.mean(z)

        # empirical standard deviation (eq 11.52c in the ML book)
        emp_var = (1 / (n*(n-1))) * np.sum([(z_val - z_hat)**2 for z_val in z])

        # computing the p-value (eq 11.53 in the ML book)
        p_val = 2 * t.cdf(x=-abs(z_hat), df=n-1, loc=0, scale=np.sqrt(emp_var))

        # computing the lower and upper bound for the confidence interval (eq 11.52a and 11.52b in the ML book)
        lower, upper = t.ppf(q=[alpha/2, 1-alpha/2], df=n-1, loc=z_hat, scale=np.sqrt(emp_var))

    elif mode == "classification":
        # computing preliminary variables
        n_12 = np.sum(fold_results["n_12"])
        n_21 = np.sum(fold_results["n_21"])
        m = np.min([n_12, n_21])
        E_theta = (n_12 - n_21) / n
        Q_numerator = n**2 * (n+1) * (E_theta + 1) * (1 - E_theta)
        Q_denominator = n * (n_12 + n_21) - (n_12 - n_21)**2
        Q = Q_numerator / Q_denominator
        f = ((E_theta + 1) * (Q - 1)) / 2
        g = ((1 - E_theta) * (Q - 1)) / 2

        # computing the p-value (eq 11.36 in the ML book)
        p_val = 2 * binom.cdf(k=m, n=n_12+n_21, p=0.5)

        # computing the lower and upper bound for the confidence interval (eq 11.35a and 11.35b in the ML book)
        lower = 2 * beta.ppf(q=alpha/2, a=f, b=g) - 1
        upper = 2 * beta.ppf(q=1-alpha/2, a=f, b=g) - 1

    return p_val, lower, upper


