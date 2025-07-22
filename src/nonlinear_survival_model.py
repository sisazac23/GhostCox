import pandas as pd
import numpy as np
from sklearn.exceptions import NotFittedError
from typing import Type, Dict, Any, Optional
import warnings

# --- Attempt to import scikit-survival ---
try:
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.util import Surv 
    
    sksurv_available = True
except ImportError:
   
    class RandomSurvivalForest: pass
    class Surv: pass
    sksurv_available = False
    warnings.warn("Package 'scikit-survival' not found. NonLinearSurvivalModel class will not function properly.", ImportWarning)
# --- End Import Handling ---

class NonLinearSurvivalModel:
    """
    A wrapper class for non-linear survival models from scikit-survival.
    Currently implements Random Survival Forest (RSF).

    Args:
        model_type (str): The type of survival model. Currently only 'rsf' supported.
        model_params: Keyword arguments passed to the underlying scikit-survival
                      model constructor. Defaults are provided for RSF.
    """
    SUPPORTED_MODELS: Dict[str, Optional[Type[Any]]] = {
        'rsf': RandomSurvivalForest if sksurv_available else None,
    }

    DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
        'rsf': {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1,
                'min_samples_split': 10, 'min_samples_leaf': 5, 'max_depth': None}
    }

    def __init__(self, model_type: str = 'rsf', **model_params):
        if not sksurv_available:
            raise ImportError("scikit-survival is required for NonLinearSurvivalModel but is not installed.")

        model_key = model_type.lower()
        model_class_ref = self.SUPPORTED_MODELS.get(model_key)

        if model_class_ref is None:
            raise ValueError(f"Unsupported model_type '{model_type}'. Choose from {list(self.SUPPORTED_MODELS.keys())}.")

        self.model_type: str = model_key
        self.model_class: Type[Any] = model_class_ref

        # Combine default params with user-provided params
        merged_params = self.DEFAULT_PARAMS.get(model_key, {}).copy()
        merged_params.update(model_params)
        self.model_params: Dict[str, Any] = merged_params

        # Instantiate the underlying model
        try:
            self.model = self.model_class(**self.model_params)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize {self.model_type.upper()} with params {self.model_params}") from e

        self._is_fitted: bool = False
        self.feature_names_in_: Optional[list] = None # Store feature names during fit

        print(f"Initialized NonLinearSurvivalModel with {self.model_type.upper()} (params: {self.model_params})")

    def fit(self, X: pd.DataFrame, y_structured: np.ndarray):
        """
        Fits the survival model to the training data.

        Args:
            X (pd.DataFrame): DataFrame of features for training.
            y_structured (np.ndarray): Structured array containing 'event' (bool)
                                       and 'time' (float) fields, typically created
                                       using sksurv.util.Surv.from_dataframe() or .from_arrays().
        """
        if not isinstance(X, pd.DataFrame):
             raise ValueError("X must be a pandas DataFrame.")
        if not sksurv_available or Surv is None: # Double check just in case
             raise ImportError("scikit-survival required for fitting.")
        # Basic check for structured array type
        if not isinstance(y_structured, np.ndarray) or not all(name in y_structured.dtype.names for name in ['event', 'time']):
             raise ValueError("y_structured must be a numpy structured array with 'event' and 'time' fields (from sksurv.util.Surv).")

        print(f"Fitting {self.model_type.upper()}...")
        try:
            # Store feature names to ensure consistency during prediction
            self.feature_names_in_ = list(X.columns)
            # Ensure X contains only numeric data before passing to sksurv model
            X_numeric = X.select_dtypes(include=np.number)
            if X_numeric.shape[1] < X.shape[1]:
                 dropped_cols = list(set(X.columns) - set(X_numeric.columns))
                 warnings.warn(f"Non-numeric columns dropped from X before fitting: {dropped_cols}", UserWarning)
            if X_numeric.isnull().any().any():
                 warnings.warn("X contains NaNs after numeric selection. Consider imputation. Fitting may fail.", UserWarning)
                 # Basic imputation (median)
                 for col in X_numeric.columns[X_numeric.isnull().any()]:
                      median_val = X_numeric[col].median()
                      X_numeric[col] = X_numeric[col].fillna(median_val)

            if X_numeric.empty:
                raise ValueError("No numeric features remaining in X after preprocessing.")

            self.model.fit(X_numeric, y_structured)
            self._is_fitted = True
            print("Fitting complete.")
        except Exception as e:
            self._is_fitted = False # Ensure state reflects failure
            raise RuntimeError(f"Failed to fit {self.model_type.upper()} model.") from e

        return self

    def predict_risk_score(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts the risk score f(x) for new data.
        For RSF, higher scores indicate higher risk (closer to log-hazard).

        Args:
            X (pd.DataFrame): DataFrame of features for prediction.

        Returns:
            np.ndarray: Array of predicted risk scores.
        """
        if not self._is_fitted:
            raise NotFittedError("This NonLinearSurvivalModel instance is not fitted yet. Call 'fit' first.")
        if not isinstance(X, pd.DataFrame):
             raise ValueError("X must be a pandas DataFrame.")
        if not sksurv_available: 
             raise ImportError("scikit-survival required for prediction.")

        # Ensure input features match training features order and presence
        if self.feature_names_in_ is None:
             raise NotFittedError("Model has been fitted but feature names were not stored.")
        if list(X.columns) != self.feature_names_in_:
             warnings.warn("Input features mismatch fitted features order/presence. Reordering/selecting columns.", UserWarning)
             try:
                 
                 X = X[self.feature_names_in_]
             except KeyError as e:
                 missing_cols = list(set(self.feature_names_in_) - set(X.columns))
                 raise ValueError(f"Input data missing columns used during fitting: {missing_cols}") from e

        # Ensure X contains only numeric data before passing to sksurv model
        X_numeric = X.select_dtypes(include=np.number)
        if X_numeric.shape[1] < X.shape[1]:
                 dropped_cols = list(set(X.columns) - set(X_numeric.columns))
                 warnings.warn(f"Non-numeric columns dropped from X before prediction: {dropped_cols}", UserWarning)
        if X_numeric.isnull().any().any():
             warnings.warn("X contains NaNs after numeric selection. Consider imputation. Prediction may fail.", UserWarning)
             # Basic imputation (median)
             for col in X_numeric.columns[X_numeric.isnull().any()]:
                 median_val = X_numeric[col].median()
                 X_numeric[col] = X_numeric[col].fillna(median_val)

        if X_numeric.shape[1] != len(self.feature_names_in_):
            # Check if only non-numeric columns existed, leading to empty df
            if X_numeric.empty and not any(X[f].dtype == np.number for f in self.feature_names_in_):
                 raise ValueError("Model was fitted on features, but input X has no numeric features corresponding to them.")
            # Or if numeric selection failed for other reasons
            raise ValueError(f"Mismatch in number of numeric features after preprocessing. Expected {len(self.feature_names_in_)}, got {X_numeric.shape[1]}.")


        try:
            # predict() for RSF returns risk scores directly
            risk_scores = self.model.predict(X_numeric)
            return risk_scores
        except Exception as e:
            raise RuntimeError(f"Failed to predict with {self.model_type.upper()} model.") from e

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted