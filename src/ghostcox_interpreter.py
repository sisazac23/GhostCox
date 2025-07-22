import pandas as pd
import numpy as np
from sklearn.exceptions import NotFittedError
from typing import Type, Dict, Any, Optional, Tuple
import warnings



class GhostCoxInterpreter:
    """
    Interprets a fitted non-linear survival model using the Ghost Variables methodology.

    Calculates variable relevance (RV_gh) based on the change in the model's
    predicted risk score when a variable is replaced by its ghost.
    Optionally calculates the Relevance Matrix V.

    Args:
        survival_model (NonLinearSurvivalModel): A fitted instance of the survival model wrapper.
        ghost_estimator (GhostVariableEstimator): An initialized instance for estimating ghost variables.
    """
    def __init__(self,
                 survival_model: 'NonLinearSurvivalModel',
                 ghost_estimator: 'GhostVariableEstimator'):

        if not hasattr(survival_model, 'predict_risk_score') or not callable(survival_model.predict_risk_score):
             raise TypeError("survival_model must have a callable 'predict_risk_score' method.")
        if not hasattr(survival_model, 'is_fitted') or not survival_model.is_fitted:
             raise NotFittedError("The provided survival_model must be fitted first.")
        if not isinstance(ghost_estimator, GhostVariableEstimator):
             raise TypeError("ghost_estimator must be an instance of GhostVariableEstimator.")

        self.survival_model = survival_model
        self.ghost_estimator = ghost_estimator
        print("Initialized GhostCoxInterpreter.")


    def calculate_relevance(self, X_test: pd.DataFrame, calculate_relevance_matrix: bool = True) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
     """
     Calculates ghost variable relevance RV_gh for each variable in X_test.

     Args:
          X_test (pd.DataFrame): The test dataset (features only).
          calculate_relevance_matrix (bool): If True, also calculates and returns the
                                             Relevance Matrix V. Defaults to True.

     Returns:
          Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
               - relevance_df: DataFrame containing relevance scores (RV_gh_numerator, RV_gh)
                              and ranks for each variable.
               - V_df: DataFrame representing the normalized Relevance Matrix V (or None if
                         calculate_relevance_matrix is False).
     """
     if not isinstance(X_test, pd.DataFrame) or X_test.empty:
          raise ValueError("X_test must be a non-empty pandas DataFrame.")
     if not self.survival_model.is_fitted:
          raise NotFittedError("The survival model within the interpreter is not fitted.")

     print("\nCalculating Ghost Variable Relevance...")
     n_samples, n_features_orig = X_test.shape
     feature_names_orig = list(X_test.columns)

     # 1. Estimate all ghost variables for the test set
     X_ghost_test = self.ghost_estimator.estimate_all_ghosts(X_test)

     # Determine successfully ghosted features
     valid_feature_names = list(X_ghost_test.columns)
     if not valid_feature_names:
          raise RuntimeError("Ghost estimation failed for all variables.")
     if len(valid_feature_names) < n_features_orig:
          missing_ghosts = set(feature_names_orig) - set(valid_feature_names)
          warnings.warn(f"Could not estimate ghosts for: {missing_ghosts}. Excluding these from relevance calculation.", UserWarning)

     # Use only the features for which ghosts were successfully estimated
     X_test_filtered = X_test[valid_feature_names]
     n_samples, n_features = X_test_filtered.shape # Update dimensions

     # 2. Get predictions from the survival model on original (filtered) test data
     try:
          f_X_test = self.survival_model.predict_risk_score(X_test_filtered)
          if f_X_test.ndim != 1 or len(f_X_test) != n_samples:
               raise ValueError(f"predict_risk_score returned unexpected shape {f_X_test.shape}")
     except Exception as e:
          raise RuntimeError("Failed to get predictions from the survival model on filtered X_test.") from e

     # --- NORMALIZATION FACTOR CALCULATION ---
     # Calculate the variance of the original predictions on the test set
     var_f_test = np.var(f_X_test)
     # Handle potential zero variance (if predictions are constant)
     if var_f_test < np.finfo(float).eps: # Use machine epsilon for safe comparison
          warnings.warn("Variance of predicted risk scores on test set is near zero. RV_gh cannot be calculated (will be NaN).", UserWarning)
          var_f_test = np.nan # Set to NaN to propagate the issue clearly
     else:
          print(f"Normalization factor Var(f_test): {var_f_test:.4f}")
     # -----------------------------------------

     relevance_results = {}
     A_matrix = np.zeros((n_samples, n_features)) # Size based on valid features

     # 3. Loop through each VALID variable to calculate relevance
     for j, var_name in enumerate(valid_feature_names):
          # print(f"... calculating relevance for {var_name} ({j+1}/{n_features})") 
          X_test_j_ghost = X_test_filtered.copy()
          try:
               
               X_test_j_ghost.loc[:, var_name] = X_ghost_test[var_name].astype(X_test_filtered[var_name].dtype, errors='ignore')
          except Exception:
               X_test_j_ghost.loc[:, var_name] = X_ghost_test[var_name]

          # Get predictions with the ghost variable substituted
          try:
               f_X_test_j_ghost = self.survival_model.predict_risk_score(X_test_j_ghost)
               if f_X_test_j_ghost.shape != f_X_test.shape:
                    raise ValueError(f"Prediction shape mismatch for ghost '{var_name}': {f_X_test_j_ghost.shape} vs {f_X_test.shape}")
          except Exception as e:
               warnings.warn(f"Could not get prediction for ghost '{var_name}': {e}. Setting relevance to NaN.", UserWarning)
               relevance_results[var_name] = {'RV_gh_numerator': np.nan, 'RV_gh': np.nan} # Store NaN for both
               A_matrix[:, j] = np.nan 
               continue

          # Calculate change and store
          prediction_change = f_X_test - f_X_test_j_ghost
          A_matrix[:, j] = prediction_change
          mean_sq_change = np.mean(prediction_change**2)

          # --- APPLY NORMALIZATION ---
          normalized_relevance = mean_sq_change / var_f_test if not np.isnan(var_f_test) else np.nan
          # --------------------------

          relevance_results[var_name] = {
               'RV_gh_numerator': mean_sq_change,
               'RV_gh': normalized_relevance # Store the normalized value
          }

     # Create DataFrame from results
     relevance_df = pd.DataFrame.from_dict(relevance_results, orient='index')

     # Filter out any NaNs that might have occurred
     nan_features = relevance_df[relevance_df['RV_gh'].isna()].index.tolist()
     if nan_features:
          warnings.warn(f"Relevance calculation resulted in NaN for: {nan_features}. Excluding them from Rank and V Matrix.", UserWarning)
          relevance_df_valid = relevance_df.dropna(subset=['RV_gh'])
          # Filter corresponding columns from A_matrix if calculating V
          valid_indices_for_V = [j for j, name in enumerate(valid_feature_names) if name in relevance_df_valid.index]
          A_matrix_valid = A_matrix[:, valid_indices_for_V]
          valid_feature_names_for_V = relevance_df_valid.index.tolist() # Update list for V
     else:
          relevance_df_valid = relevance_df # Use the full df if no NaNs
          A_matrix_valid = A_matrix
          valid_feature_names_for_V = valid_feature_names

     if relevance_df_valid.empty:
          # Add Rank column even if empty to avoid errors later, though it will be empty
          relevance_df['Rank'] = np.nan
          warnings.warn("Relevance calculation failed or resulted in NaN for all variables. Cannot rank or calculate V matrix.", RuntimeWarning)
     else:
          # Calculate Rank based on the normalized RV_gh for valid features
          relevance_df['Rank'] = relevance_df_valid['RV_gh'].rank(ascending=False).astype(int)
          # Add NaN ranks for features that had NaN relevance
          relevance_df['Rank'] = relevance_df['Rank'].reindex(relevance_df.index)


     # # 4. Calculate Relevance Matrix V (optional)
     # # --- Calculate V as the sample covariance matrix of ghost effects (A_matrix_valid) ---
     #        # A_matrix_valid has shape (n_samples, n_valid_features_for_V)
     #        # We want the covariance between the columns (variables).
     # if A_matrix_valid.shape[0] > 1: # np.cov requires at least 2 observations for sample covariance (ddof=1 default)
     #     # rowvar=False means each column is a variable, each row is an observation
     #     V_matrix = np.cov(A_matrix_valid, rowvar=False) 
     # elif A_matrix_valid.shape[0] == 1 and A_matrix_valid.shape[1] > 0: # Single observation case
     #      # For a single observation, covariance is typically undefined or zero. 
     #      # np.cov with ddof=0 would give 0. With ddof=1 (default) it gives NaN or error.
     #      # Let's return a matrix of zeros for consistency, or NaNs.
     #      # D&P's R code cov(A) would likely produce NaNs or errors if A has 1 row.
     #      num_valid_features_for_V = A_matrix_valid.shape[1]
     #      V_matrix = np.full((num_valid_features_for_V, num_valid_features_for_V), 0.0) # Or np.nan
     #      warnings.warn("Covariance matrix V is ill-defined with only one observation in A_matrix_valid; set to zeros/NaNs.", UserWarning)
     # else: # No data or no features
     #      num_valid_features_for_V = A_matrix_valid.shape[1] if A_matrix_valid.ndim == 2 else 0
     #      V_matrix = np.empty((num_valid_features_for_V, num_valid_features_for_V)) # Empty or NaN matrix
     #      V_matrix[:] = np.nan
     #      warnings.warn("Not enough data or features in A_matrix_valid to calculate covariance matrix V.", UserWarning)
     
     # # ------------------------------------------------------------------------------------
     # V_df = pd.DataFrame(V_matrix, index=valid_feature_names_for_V, columns=valid_feature_names_for_V)
     # 4. Calculate Relevance Matrix V (optional)
     V_df = None # Initialize V_df to None
     if calculate_relevance_matrix:
         if A_matrix_valid.shape[1] > 0 : # If there are valid features to calculate V for
             n_samples_for_V = A_matrix_valid.shape[0]
             n_features_for_V = A_matrix_valid.shape[1]

             if n_samples_for_V > 0:
                 # Calculate M = (1/n_samples) * A'A
                 # This M is the matrix of mean raw cross-products of prediction changes.
                 # Its diagonal M_jj is R_j^num for variable j.
                 M_matrix = (A_matrix_valid.T @ A_matrix_valid) / n_samples_for_V

                 # Now, normalize M_matrix by var_f_test to get the final V
                 if not np.isnan(var_f_test) and var_f_test > np.finfo(float).eps:
                     V_matrix_final = M_matrix / var_f_test
                 else:
                     # If var_f_test is invalid, V components will be NaN
                     V_matrix_final = np.full((n_features_for_V, n_features_for_V), np.nan)
                     warnings.warn("Normalization factor Var(f_test) is invalid for V matrix; V set to NaNs.", UserWarning)

                 V_df = pd.DataFrame(V_matrix_final, index=valid_feature_names_for_V, columns=valid_feature_names_for_V)
                 print("Relevance Matrix V calculated.")

             else: # n_samples_for_V is 0, but valid_feature_names_for_V might exist
                 V_matrix_final = np.full((n_features_for_V, n_features_for_V), np.nan)
                 warnings.warn("Not enough samples in A_matrix_valid (0 samples) to calculate Relevance Matrix V; V set to NaNs.", UserWarning)
                 V_df = pd.DataFrame(V_matrix_final, index=valid_feature_names_for_V, columns=valid_feature_names_for_V)

         else: # No valid features for V
             warnings.warn("No valid features to calculate Relevance Matrix V.", UserWarning)
             # V_df remains None, or you could assign an empty DataFrame:
             # V_df = pd.DataFrame() 

     print("Relevance calculation complete.")
          # Return the potentially NaN-containing relevance_df and the potentially None V_df
     return relevance_df.sort_values('Rank', na_position='last'), V_df

