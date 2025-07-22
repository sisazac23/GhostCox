import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, RegressorMixin 
from sklearn.exceptions import NotFittedError 


try:
    from pygam import LinearGAM
except ImportError:
    LinearGAM = None

from typing import Type, Dict, Any, List, Union, Optional, Tuple 
import warnings
import joblib
import inspect

class GhostVariableEstimator:
    SUPPORTED_ESTIMATORS: Dict[str, Optional[Type[BaseEstimator]]] = {
        'lm': LinearRegression,
        'rf': RandomForestRegressor,
        'gam': LinearGAM
    }

    DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
        'lm': {'n_jobs': None},
        'rf': {'n_estimators': 100, 'random_state': 42, 'n_jobs': 1, 'max_depth': 10, 'min_samples_leaf': 5},
        'gam': {'n_splines': 10, 'max_iter': 100, 'tol': 1e-3}
    }

    def __init__(self, estimator_type: str = 'rf', n_jobs: Optional[int] = 1, **estimator_params):
        estimator_key = estimator_type.lower()
        if estimator_key not in self.SUPPORTED_ESTIMATORS:
            raise ValueError(f"Invalid estimator_type '{estimator_type}'. Choose from {list(k for k,v in self.SUPPORTED_ESTIMATORS.items() if v is not None)}.")
        if self.SUPPORTED_ESTIMATORS[estimator_key] is None:
            print(f"Optional dependency for estimator type '{estimator_key}' not found. Please install it (e.g., pip install pygam).")
            raise ImportError(f"Estimator type '{estimator_key}' requires an optional dependency that is not installed.")

        self.estimator_key: str = estimator_key
        self.estimator_class: Type[BaseEstimator] = self.SUPPORTED_ESTIMATORS[estimator_key]

        merged_params = self.DEFAULT_PARAMS.get(estimator_key, {}).copy()
        merged_params.update(estimator_params)
        self.estimator_params: Dict[str, Any] = merged_params
        self.n_jobs = n_jobs
        self._fitted_estimators: Dict[str, BaseEstimator] = {} 

        print(f"Initialized GhostVariableEstimator with {self.estimator_key.upper()} (params: {self.estimator_params}, requested n_jobs={self.n_jobs})")
        if self.n_jobs != 1:
            print("Note: Using n_jobs != 1 may cause serialization errors depending on the environment/estimators.")


    def _get_estimator_instance(self) -> BaseEstimator: 
        """Instantiates a new underlying estimator model with configured parameters."""
        sig = inspect.signature(self.estimator_class.__init__)
        valid_keys = {p.name for p in sig.parameters.values() if p.kind == p.POSITIONAL_OR_KEYWORD or p.kind == p.KEYWORD_ONLY}
        if 'self' not in valid_keys and any(p.name == 'self' for p in sig.parameters.values()):
            valid_keys.add('self') 
        init_params = {k: v for k, v in self.estimator_params.items() if k in valid_keys}
        try:
            return self.estimator_class(**init_params)
        except TypeError as e:
            warnings.warn(f"Error initializing {self.estimator_key.upper()} with filtered params {init_params}: {e}. Trying default.", UserWarning)
            try:
                return self.estimator_class()
            except Exception as e_def:
                raise RuntimeError(f"Failed to initialize {self.estimator_key.upper()} with both provided and default params.") from e_def

    def _estimate_single_ghost_job(self, X_df: pd.DataFrame, target_var_name: str) -> Optional[pd.Series]: 
        current_job_id = f"{target_var_name} (using {self.estimator_key})"
        try:
            if target_var_name not in X_df.columns:
                warnings.warn(f"[{current_job_id}] Target variable not found in DataFrame.", UserWarning)
                return None
            if X_df.shape[1] <= 1:
                warnings.warn(f"[{current_job_id}] Cannot estimate ghost with <= 1 feature. Returning original.", UserWarning)
                return X_df[target_var_name].astype(float)

            target_series = X_df[target_var_name]
            feature_df = X_df.drop(columns=[target_var_name])

            if not pd.api.types.is_numeric_dtype(target_series):
                warnings.warn(f"[{current_job_id}] Target is not numeric. Cannot estimate ghost. Returning original.", UserWarning)
                return target_series.astype(float)

            if self.estimator_key in ['lm', 'rf', 'gam']:
                numeric_features = feature_df.select_dtypes(include=np.number)
                if numeric_features.empty:
                    warnings.warn(f"[{current_job_id}] No numeric features found for {self.estimator_key}. Returning original.", UserWarning)
                    return target_series.astype(float)
                if numeric_features.shape[1] < feature_df.shape[1]:
                    warnings.warn(f"[{current_job_id}] Non-numeric features dropped for {self.estimator_key}.", UserWarning)
                feature_df_processed = numeric_features
            else:
                feature_df_processed = feature_df

            if feature_df_processed.isnull().any().any():
                warnings.warn(f"[{current_job_id}] Features contain NaNs after selection. Applying median imputation.", UserWarning)
                for col in feature_df_processed.columns[feature_df_processed.isnull().any()]: # Iterate only over columns with NaNs
                    median_val = feature_df_processed[col].median()
                    feature_df_processed.loc[:, col] = feature_df_processed[col].fillna(median_val)

            estimator = self._get_estimator_instance()
            estimator.fit(feature_df_processed.values, target_series.values)
            Z_ghost = estimator.predict(feature_df_processed.values)
            self._fitted_estimators[target_var_name] = estimator 
            return pd.Series(Z_ghost.astype(float), index=X_df.index, name=target_var_name)
        except Exception as e:
            warnings.warn(f"[{current_job_id}] Error during estimation: {e}. Returning None.", UserWarning)
            import traceback 
            print(f"Traceback for {current_job_id}:\n{traceback.format_exc()}")
            return None

    def estimate_all_ghosts(self, X_data: pd.DataFrame) -> pd.DataFrame: 
        if not isinstance(X_data, pd.DataFrame) or X_data.empty:
            raise ValueError("Input must be a non-empty pandas DataFrame.")
        if X_data.isnull().any().any():
            warnings.warn("Input data contains NaNs. Consider imputation before calling estimate_all_ghosts.", UserWarning)

        print(f"\nEstimating all ghost variables using {self.estimator_key.upper()}...")
        target_vars = list(X_data.columns)
        verbosity = 5 if self.n_jobs != 1 else 0
        try:
            
            with joblib.Parallel(n_jobs=self.n_jobs, backend='loky', verbose=verbosity) as parallel:
                results = parallel(
                    joblib.delayed(self._estimate_single_ghost_job)(X_data.copy(), var_name)
                    for var_name in target_vars
                )
        except Exception as e:
            warnings.warn(f"Joblib parallel execution failed: {e}. Attempting serial execution.", RuntimeWarning)
            results = [self._estimate_single_ghost_job(X_data.copy(), var_name) for var_name in target_vars]

        valid_results_map = {res.name: res for res in results if res is not None and hasattr(res, 'name')}

        if not valid_results_map:
            raise RuntimeError("All ghost variable estimations failed. Check warnings.")

        successful_cols = [col for col in target_vars if col in valid_results_map]
        if not successful_cols: 
             raise RuntimeError("No columns successfully processed for ghost estimation.")
        ghost_df = pd.concat([valid_results_map[col] for col in successful_cols], axis=1)

        failed_cols = set(target_vars) - set(successful_cols)
        if failed_cols:
            warnings.warn(f"Ghost variable estimation failed for: {sorted(list(failed_cols))}. Check previous warnings.", UserWarning)

        print(f"Ghost variable estimation complete. Successfully estimated for {len(successful_cols)} out of {len(target_vars)} variables.")
        return ghost_df

    
    def predict_single_variable_ghost_on_new_data(self, X_new_df: pd.DataFrame, target_var_name: str) -> Optional[pd.Series]: 
        if target_var_name not in self._fitted_estimators:
            warnings.warn(f"No pre-fitted estimator found for target variable '{target_var_name}'. Returning None.", UserWarning)
            return None
        if target_var_name not in X_new_df.columns:
            warnings.warn(f"Target variable '{target_var_name}' not found in X_new_df. Returning None.", UserWarning)
            return None

        estimator = self._fitted_estimators[target_var_name]
        if X_new_df.shape[1] <= 1:
            return X_new_df[target_var_name].astype(float)

        feature_df = X_new_df.drop(columns=[target_var_name])
        
        if self.estimator_key in ['lm', 'rf', 'gam']:
            numeric_features = feature_df.select_dtypes(include=np.number)
            if numeric_features.empty:
                warnings.warn(f"No numeric features in X_new_df for '{target_var_name}' using {self.estimator_key}. Returning original.", UserWarning)
                return X_new_df[target_var_name].astype(float)
            if numeric_features.shape[1] < feature_df.shape[1]:
                warnings.warn(f"Non-numeric features dropped from X_new_df for predicting '{target_var_name}'.", UserWarning)
            feature_df_processed = numeric_features
        else:
            feature_df_processed = feature_df

        if feature_df_processed.isnull().any().any():
            warnings.warn(f"Features in X_new_df for '{target_var_name}' contain NaNs. Applying median imputation.", UserWarning)
            
            for col in feature_df_processed.columns[feature_df_processed.isnull().any()]:
                median_val = feature_df_processed[col].median()
                feature_df_processed.loc[:, col] = feature_df_processed[col].fillna(median_val)
        
        try:
            Z_ghost_new = estimator.predict(feature_df_processed.values)
            return pd.Series(Z_ghost_new.astype(float), index=X_new_df.index, name=target_var_name)
        except Exception as e:
            warnings.warn(f"Error predicting ghost for '{target_var_name}' on new data: {e}. Returning None.", UserWarning)
            return None

    @property 
    def is_fitted(self) -> bool:
        return bool(self._fitted_estimators)