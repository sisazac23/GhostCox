import pandas as pd
import numpy as np
from typing import Type, Dict, Any, List, Union, Optional, Tuple, Callable
import warnings

try:
    from sksurv.util import Surv # Required for scikit-survival format
    sksurv_available = True
except ImportError:
    Surv = None 
    sksurv_available = False
    warnings.warn("Package 'scikit-survival' not found. Data generation will not produce sksurv structured arrays.", ImportWarning)



import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Callable, Union, Tuple
import warnings


try:
    from sksurv.util import Surv
    sksurv_available = True
except ImportError:
    class Surv: pass # Dummy
    sksurv_available = False

def generate_uncorrelated_features(n_samples: int, n_features: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generates independent standard normal features."""
    data = rng.normal(0, 1, size=(n_samples, n_features))
    feature_names = [f'X{i+1}' for i in range(n_features)]
    return pd.DataFrame(data, columns=feature_names)

def generate_correlated_features(n_samples: int, n_features: int, rng: np.random.Generator, corr_matrix: np.ndarray) -> pd.DataFrame:
    """Generates multivariate normal features with a specified correlation matrix."""
    if corr_matrix.shape != (n_features, n_features):
        raise ValueError("Correlation matrix dimensions must match n_features.")
    # Use Cholesky decomposition to generate correlated data
    try:
        L = np.linalg.cholesky(corr_matrix)
    except np.linalg.LinAlgError:
         raise ValueError("Correlation matrix is not positive definite!")
    uncorrelated_data = rng.normal(0, 1, size=(n_samples, n_features))
    correlated_data = uncorrelated_data @ L.T # Apply transformation
    feature_names = [f'X{i+1}' for i in range(n_features)]
    return pd.DataFrame(correlated_data, columns=feature_names)

# Define the non-linear function globally too
def nonlinear_predictor_func(df: pd.DataFrame) -> np.ndarray:
    """Example non-linear function f(x)."""
    required_cols = ['X1', 'X2', 'X3']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame missing required columns for nonlinear_predictor_func: {required_cols}")
    score = 1.0 * df['X1'] + np.sin(np.pi * df['X2']) + 0.5 * (df['X3']**2)
    return score.values


import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Callable, Union, Tuple
import warnings


try:
    from sksurv.util import Surv
    sksurv_available = True
except ImportError:
    class Surv: pass # Dummy
    sksurv_available = False


class SurvivalDataGenerator:
    def __init__(self,
                 n_samples: int = 500,
                 n_features: int = 3,
                 # --- NEW ARGUMENTS ---
                 feature_generation_type: str = 'uncorrelated', # 'uncorrelated' or 'correlated'
                 corr_matrix: Optional[np.ndarray] = None, # Required if type is 'correlated'
                 # --- END NEW ARGUMENTS ---
                 beta_coeffs: Optional[List[float]] = None,
                 nonlinear_func: Optional[Callable[[pd.DataFrame], np.ndarray]] = None,
                 baseline_hazard_lambda: float = 0.01,
                 censoring_dist_lambda: float = 0.002,
                 contamination_prop: float = 0.0,
                 random_state: Optional[int] = None):

        # --- VALIDATION FOR NEW ARGUMENTS ---
        self.feature_generation_type = feature_generation_type.lower()
        if self.feature_generation_type not in ['uncorrelated', 'correlated']:
            raise ValueError("feature_generation_type must be 'uncorrelated' or 'correlated'")
        if self.feature_generation_type == 'correlated' and corr_matrix is None:
            raise ValueError("corr_matrix must be provided when feature_generation_type is 'correlated'")
        if self.feature_generation_type == 'correlated' and corr_matrix.shape != (n_features, n_features):
             raise ValueError(f"corr_matrix shape {corr_matrix.shape} does not match n_features {n_features}")
        self.corr_matrix = corr_matrix
        # --- END VALIDATION ---

        if beta_coeffs is not None and len(beta_coeffs) != n_features:
             raise ValueError("Length of beta_coeffs must match n_features.")
        
        elif beta_coeffs is None and nonlinear_func is None:
             warnings.warn("Neither beta_coeffs nor nonlinear_func provided. Using f(X) = sum(X_i).", UserWarning)
             self.nonlinear_func = lambda x_df: x_df.sum(axis=1).values # Default simple sum
             self.beta_coeffs = None
        elif beta_coeffs is not None and nonlinear_func is not None:
             warnings.warn("Both beta_coeffs and nonlinear_func provided. Using nonlinear_func.", UserWarning)
             self.beta_coeffs = None
             self.nonlinear_func = nonlinear_func
        elif beta_coeffs is not None:
             self.beta_coeffs = np.array(beta_coeffs)
             self.nonlinear_func = None
        else: # nonlinear_func is provided, beta_coeffs is None
             self.beta_coeffs = None
             self.nonlinear_func = nonlinear_func


        self.n_samples = n_samples
        self.n_features = n_features
        self.feature_names = [f'X{i+1}' for i in range(n_features)]
        
        self.beta_coeffs = np.array(beta_coeffs) if beta_coeffs is not None else None
        self.nonlinear_func = nonlinear_func
        self.baseline_hazard_lambda = baseline_hazard_lambda
        self.censoring_dist_lambda = censoring_dist_lambda
        self.contamination_prop = contamination_prop
        self.rng = np.random.default_rng(random_state)

        print(f"Initialized SurvivalDataGenerator (n={n_samples}, p={n_features}, type='{self.feature_generation_type}', alpha={contamination_prop})")


    def _generate_features(self) -> pd.DataFrame:
        """ Generates features based on the specified type. """
        try:
            if self.feature_generation_type == 'uncorrelated':
                # Ensure generate_uncorrelated_features is defined globally
                features_df = generate_uncorrelated_features(self.n_samples, self.n_features, self.rng)
            elif self.feature_generation_type == 'correlated':
                # Ensure generate_correlated_features is defined globally
                if self.corr_matrix is None: 
                    raise ValueError("Correlation matrix is missing for 'correlated' type.")
                features_df = generate_correlated_features(self.n_samples, self.n_features, self.rng, self.corr_matrix)
            else:
                
                raise ValueError(f"Unknown feature_generation_type: {self.feature_generation_type}")

        except NameError as e:
             raise NameError(f"Helper function for '{self.feature_generation_type}' features not defined globally: {e}")
        except Exception as e:
             raise RuntimeError(f"Error during feature generation for type '{self.feature_generation_type}'.") from e

        
        if features_df.shape != (self.n_samples, self.n_features):
             raise ValueError(f"Feature generation produced shape {features_df.shape}, expected ({self.n_samples}, {self.n_features}).")
        features_df.columns = self.feature_names
        return features_df[self.feature_names] # Return in defined order

    # --- Methods _calculate_predictor_score  ---
    
    def _calculate_predictor_score(self, X: pd.DataFrame) -> np.ndarray:
        """ Calculates the linear or non-linear predictor score f(X) or beta'X. """
        if self.nonlinear_func:
            try:
                 score = self.nonlinear_func(X)
                 if not isinstance(score, np.ndarray) or score.ndim != 1 or score.shape[0] != X.shape[0]:
                      raise ValueError(f"nonlinear_func must return a 1D numpy array of length {X.shape[0]}")
                 return score
            except Exception as e:
                 raise ValueError("Error executing nonlinear_func.") from e
        elif self.beta_coeffs is not None:
             if not all(col in X.columns for col in self.feature_names):
                  raise ValueError("Input DataFrame X is missing expected feature columns for beta calculation.")
             return X[self.feature_names].values @ self.beta_coeffs
        else:
             warnings.warn("Neither beta_coeffs nor nonlinear_func provided. Using f(X) = sum(X_i).", UserWarning)
             return X[self.feature_names].sum(axis=1).values

    def generate(self, return_sksurv_array: bool = True) -> Union[Tuple[pd.DataFrame, np.ndarray], Tuple[pd.DataFrame, pd.Series, pd.Series]]:
        """ Generates the survival dataset. """
        if return_sksurv_array and not sksurv_available:
             warnings.warn("scikit-survival not found. Returning time and event as separate Series.", ImportWarning)
             return_sksurv_array = False

        
        X = self._generate_features()
        predictor_score = self._calculate_predictor_score(X)

        # Generate latent event times
        rate = self.baseline_hazard_lambda * np.exp(predictor_score)
        rate = np.maximum(rate, 1e-12)
        scale = 1.0 / rate
        T_latent = self.rng.exponential(scale=scale, size=self.n_samples)

        # Contamination
        
        outlier_indices = np.array([], dtype=int)
        if self.contamination_prop > 0:
            n_outliers = int(np.round(self.contamination_prop * self.n_samples))
            if n_outliers > 0 and n_outliers < self.n_samples:
                outlier_indices = self.rng.choice(self.n_samples, n_outliers, replace=False)
                p01 = np.percentile(T_latent, 1)
                p99 = np.percentile(T_latent, 99)
                max_T = np.max(T_latent)
                outlier_times_early = self.rng.uniform(0, max(p01 * 0.9, 1e-6), n_outliers)
                late_lower_bound = p99
                late_upper_bound = max(p99 * 1.01, max_T * 1.1)
                if late_upper_bound <= late_lower_bound:
                    late_upper_bound = late_lower_bound + 1e-6 * (late_lower_bound + 1.0)
                outlier_times_late = self.rng.uniform(late_lower_bound, late_upper_bound, n_outliers)
                u_i = self.rng.binomial(1, 0.5, n_outliers)
                T_latent[outlier_indices] = np.where(u_i == 1, outlier_times_early, outlier_times_late)


        # Generate censoring times
        if self.censoring_dist_lambda <= 0:
             T_cens = np.full(self.n_samples, np.inf)
             # print("censoring_dist_lambda <= 0 implies effectively no censoring.")
        else:
             T_cens = self.rng.exponential(scale=1.0 / self.censoring_dist_lambda, size=self.n_samples)

        # Observed time and event status
        E_bool = (T_latent <= T_cens)
        T_obs = np.minimum(T_latent, T_cens)

        # Handle potential NaNs or Infs
        valid_idx = np.isfinite(T_obs) & np.isfinite(predictor_score) & ~np.isnan(T_obs) & ~np.isnan(predictor_score)
        if not np.all(valid_idx):
             n_removed = self.n_samples - np.sum(valid_idx)
             warnings.warn(f"Removed {n_removed} samples due to non-finite time or predictor score values.", UserWarning)
             X = X.loc[valid_idx].reset_index(drop=True)
             T_obs = T_obs[valid_idx]
             E_bool = E_bool[valid_idx]
             if X.empty:
                  raise ValueError("All samples removed due to invalid values during generation.")

        actual_censoring_prop = 1.0 - E_bool.mean()
        # print(f"Data generation complete. Final n={X.shape[0]}. Actual censoring: {actual_censoring_prop:.2f}") 

        if return_sksurv_array:
            y_structured = Surv.from_arrays(event=E_bool, time=T_obs)
            return X, y_structured
        else:
            return X, pd.Series(T_obs, name='time'), pd.Series(E_bool, name='event')

