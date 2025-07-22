# %%writefile experiment_runner.py # Use if in a notebook

import pandas as pd
import numpy as np
from time import time
from typing import List, Dict, Any, Optional, Callable, Tuple
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import inspect
from functools import partial # For the test statistic function

# --- Attempt to import HRT and related components ---
try:
    from pyhrt import hrt # Assuming hrt.py is accessible
    hrt_available = True
except ImportError as e:
    warnings.warn(f"HRT modules (e.g., hrt.py) not found. HRT analysis will be skipped. Error: {e}", ImportWarning)
    hrt_available = False


# from survival_data_generator import SurvivalDataGenerator
# from nonlinear_survival_model import NonLinearSurvivalModel
# from ghost_variables import GhostVariableEstimator
# from ghost_cox_interpreter import GhostCoxInterpreter # Make sure this is correctly named

try:
    from sksurv.metrics import concordance_index_censored
    from sksurv.util import Surv
    sksurv_available = True
except ImportError:
    concordance_index_censored = None # Define for type hinting even if not available
    Surv = None
    sksurv_available = False
    warnings.warn("scikit-survival not found. C-index calculation might fail.", ImportWarning)

# --- Test Statistic Function for HRT (using C-Index Loss) ---
def c_index_loss_for_hrt(X_test_perturbed_np: np.ndarray, 
                         feature_names_list: list, 
                         fixed_rsf_model: 'NonLinearSurvivalModel', # Forward reference
                         fixed_y_test_sksurv: np.ndarray) -> float:
    """
    Test statistic for HRT:  C-Index 
    """
    # Convert NumPy array back to DataFrame with correct column names if your model expects it
    X_test_perturbed_df = pd.DataFrame(X_test_perturbed_np, columns=feature_names_list)
    
    try:
        risk_scores = fixed_rsf_model.predict_risk_score(X_test_perturbed_df)
        event_indicator = fixed_y_test_sksurv['event'].astype(bool)
        event_time = fixed_y_test_sksurv['time']
        
        c_index_val, _, _, _, _ = concordance_index_censored(
            event_indicator, 
            event_time, 
            risk_scores
        )
        if np.isnan(c_index_val):
            return 0 # Max loss for NaN C-index (worst outcome)
        return c_index_val
    except Exception as e:
        # warnings.warn(f"C-index calculation failed during HRT tstat_fn: {e}") 
        return 0 # Max loss if C-index calculation fails

class ExperimentRunner:
    def __init__(self,
                 scenarios: List[Dict[str, Any]],
                 n_replicates: int = 1000, 
                 test_size: float = 0.3,
                 random_state_base: Optional[int] = 42):

        if not sksurv_available:
            pass 

        self.scenarios = scenarios
        self.n_replicates = n_replicates
        self.test_size = test_size
        self.random_state_base = random_state_base
        self.results = []
        self._summary_df = None
        # Check for HRT availability at init
        self.hrt_is_available = hrt_available 
        if not self.hrt_is_available:
             print("Warning: HRT modules not found. HRT analysis will be skipped for all scenarios.")
        print(f"Initialized ExperimentRunner with {len(scenarios)} scenarios and {n_replicates} replicates.")

    def _run_hrt_for_scenario_replicate(self, 
                                       X_train_pd: pd.DataFrame, 
                                       X_test_pd: pd.DataFrame, 
                                       y_test_sksurv: np.ndarray, 
                                       rsf_model: 'NonLinearSurvivalModel',
                                       feature_names: list,
                                       hrt_params: dict) -> Dict[str, float]:
        """
        Runs HRT for all features for a single replicate.
        hrt_params: Dict containing parameters like 'nperms', 'nbootstraps', 'nfolds'.
        """
        if not self.hrt_is_available:
            return {fname: np.nan for fname in feature_names}

        hrt_p_values_dict = {}
        X_train_np = X_train_pd.to_numpy()
        X_test_np = X_test_pd.to_numpy()

        print(f"    Starting HRT analysis for {len(feature_names)} features...")
        for feature_idx, feature_name in enumerate(feature_names):
            # print(f"      HRT for feature: {feature_name} ({feature_idx + 1}/{len(feature_names)})") 
            
            # Create the test statistic function with fixed model and y_test for this replicate
            current_tstat_fn = partial(c_index_loss_for_hrt,
                                       feature_names_list=feature_names,
                                       fixed_rsf_model=rsf_model,
                                       fixed_y_test_sksurv=y_test_sksurv)
            try:
                nperms = hrt_params.get('nperms', 1000) 
                nbootstraps = hrt_params.get('nbootstraps', 100) 
                nfolds = hrt_params.get('nfolds', 5) 
                verbose_hrt = hrt_params.get('verbose_hrt', False)


                # Call the imported hrt function
                hrt_output_dict = hrt( 
                    feature=feature_idx,
                    tstat_fn=current_tstat_fn,
                    X=X_train_np,      # For training P(X_j|X_-j) via calibrate_continuous
                    X_test=X_test_np,  # For evaluating tstat_fn with the main RSF model
                    nperms=nperms,
                    nbootstraps=nbootstraps,
                    nfolds=nfolds,
                    verbose=verbose_hrt 
                )
                p_value = hrt_output_dict.get('p_value', np.nan)
            except Exception as e_hrt:
                print(f"      ERROR running HRT for {feature_name}: {e_hrt}")
                p_value = np.nan
                
            hrt_p_values_dict[feature_name] = p_value
            # print(f"      P-value for {feature_name}: {p_value:.4f}") 
        
        print(f"    HRT analysis completed.")
        return hrt_p_values_dict

    def _run_single_replication(self, scenario_config: Dict[str, Any], replicate_id: int) -> Optional[Dict[str, Any]]:
        scenario_name = scenario_config.get('name', f'Scenario_{replicate_id}')
        run_seed = None
        if self.random_state_base is not None:
            scenario_index = self.scenarios.index(scenario_config) if scenario_config in self.scenarios else 0
            run_seed = self.random_state_base + (scenario_index * self.n_replicates) + replicate_id
        
        print(f"\n--- Running: {scenario_name}, Replicate: {replicate_id+1}/{self.n_replicates} (Seed: {run_seed}) ---")
        start_time = time()

        # --- Ensure class definitions are available ---
        required_classes = ['SurvivalDataGenerator', 'NonLinearSurvivalModel', 
                            'GhostVariableEstimator', 'GhostCoxInterpreter'] 
        for cls_name in required_classes:
            if cls_name not in globals() and cls_name not in locals():
                try:
                    pass 
                except ImportError:
                     raise NameError(f"Class '{cls_name}' is not defined. Ensure it's imported.")


        try:
            # 1. Generate Data
            data_params = scenario_config.get('data_params', {}).copy()
            if 'random_state' in inspect.signature(SurvivalDataGenerator.__init__).parameters:
                data_params['random_state'] = run_seed
            generator = SurvivalDataGenerator(**data_params)
            X_orig_df, y_sksurv_orig = generator.generate(return_sksurv_array=True) 
            feature_names_orig = list(X_orig_df.columns)


            if y_sksurv_orig['event'].sum() < 5: # min_events_required
                warnings.warn(f"[{scenario_name} Rep {replicate_id+1}] Insufficient events. Skipping.", UserWarning)
                return None

            # 2. Split Data (ensure X_train/X_test can be passed as DataFrames)
            X_train_pd, X_test_pd, y_train_sksurv, y_test_sksurv = train_test_split(
                X_orig_df, y_sksurv_orig, test_size=self.test_size, random_state=run_seed, 
                stratify=y_sksurv_orig['event'] if y_sksurv_orig['event'].sum() > 1 else None
            )
            if X_train_pd.empty or X_test_pd.empty or y_test_sksurv['event'].sum() < 1:
                warnings.warn(f"[{scenario_name} Rep {replicate_id+1}] Train/test split issue. Skipping.", UserWarning)
                return None

            # 3. Fit Survival Model
            model_params = scenario_config.get('model_params', {}).copy()
            if 'random_state' in inspect.signature(NonLinearSurvivalModel.__init__).parameters:
                model_params['random_state'] = run_seed
            rsf_model = NonLinearSurvivalModel(**model_params) 
            rsf_model.fit(X_train_pd, y_train_sksurv)

            # 4. Calculate C-Index on Test Set (for overall model performance)
            risk_scores_main_model = rsf_model.predict_risk_score(X_test_pd)
            c_index_main_model_tuple = concordance_index_censored(
                y_test_sksurv['event'].astype(bool), y_test_sksurv['time'], risk_scores_main_model
            )
            c_index_main = c_index_main_model_tuple[0] if c_index_main_model_tuple else np.nan
            
            # Initialize results dictionary
            current_result = {
                'scenario_name': scenario_name,
                'replicate_id': replicate_id,
                'random_seed': run_seed,
                'c_index': c_index_main,
                'n_features': len(feature_names_orig),
                'n_train': X_train_pd.shape[0],
                'n_test': X_test_pd.shape[0],
                'survival_model_type': rsf_model.model_type if hasattr(rsf_model, 'model_type') else 'N/A',
                'error': None
            }

            # --- 5. GhostCox Interpretation ---
            run_ghostcox = scenario_config.get('run_ghostcox', True) # Default to running GhostCox
            if run_ghostcox:
                ghost_params = scenario_config.get('ghost_params', {}).copy()
                if 'random_state' in inspect.signature(GhostVariableEstimator.__init__).parameters:
                    ghost_params['random_state'] = run_seed
                ghost_estimator = GhostVariableEstimator(**ghost_params)
                
                interpreter_params = scenario_config.get('interpreter_params', {}).copy()
                interpreter = GhostCoxInterpreter(survival_model=rsf_model, ghost_estimator=ghost_estimator)
                # Ensure X_test_pd is passed if interpreter expects DataFrame
                gc_relevance_df, gc_V_matrix_df = interpreter.calculate_relevance(
                    X_test_pd, 
                    calculate_relevance_matrix=interpreter_params.get('calculate_relevance_matrix', True)
                )
                current_result['ghost_cox_relevance_df'] = gc_relevance_df
                current_result['V_matrix_df'] = gc_V_matrix_df 
                current_result['ghost_estimator_type'] = ghost_estimator.estimator_key if hasattr(ghost_estimator, 'estimator_key') else 'N/A'
            else:
                current_result['ghost_cox_relevance_df'] = None
                current_result['V_matrix_df'] = None
                current_result['ghost_estimator_type'] = 'N/A'


            # --- 6. HRT Analysis ---
            run_hrt_flag = scenario_config.get('run_hrt', True) 
            if run_hrt_flag and self.hrt_is_available:
                hrt_specific_params = scenario_config.get('hrt_params', {})
                hrt_p_values = self._run_hrt_for_scenario_replicate(
                    X_train_pd, X_test_pd, y_test_sksurv, rsf_model, 
                    feature_names_orig, hrt_specific_params
                )
                current_result['hrt_p_values'] = hrt_p_values
            else:
                current_result['hrt_p_values'] = {fname: np.nan for fname in feature_names_orig}
            
            current_result['duration_sec'] = time() - start_time
            print(f"--- Completed: {scenario_name}, Replicate: {replicate_id+1} (C-Index: {c_index_main:.4f}, Time: {current_result['duration_sec']:.2f}s) ---")
            return current_result

        except Exception as e:
            print(f"--- FAILED: {scenario_name}, Replicate: {replicate_id+1} ---")
            import traceback
            traceback.print_exc()
            ghost_est_type = scenario_config.get('ghost_params', {}).get('estimator_type', 'N/A')
            surv_model_type = scenario_config.get('model_params', {}).get('model_type', 'N/A')
            return {
                'scenario_name': scenario_name, 'replicate_id': replicate_id, 'random_seed': run_seed,
                'error': f"{type(e).__name__}: {e}", 'duration_sec': time() - start_time,
                'c_index': np.nan, 'ghost_cox_relevance_df': None, 'V_matrix_df': None,
                'hrt_p_values': {fname: np.nan for fname in feature_names_orig if 'feature_names_orig' in locals()}, 
                'ghost_estimator_type': ghost_est_type, 'survival_model_type': surv_model_type,
            }

    def run_experiment(self):
        print(f"\n===== Starting Experiment ({len(self.scenarios)} scenarios, {self.n_replicates} replicates each) =====")
        self.results = []
        self._summary_df = None # Clear cached summary
        total_start_time = time()

        for i, scenario_config in enumerate(self.scenarios):
            
            print(f"\n===== Running Scenario {i+1}/{len(self.scenarios)}: {scenario_config.get('name', 'Unnamed')} =====")
            for rep in range(self.n_replicates):
                result = self._run_single_replication(scenario_config, rep)
                if result:
                    self.results.append(result)

        total_duration = time() - total_start_time
        print(f"\n===== Experiment Finished (Total Time: {total_duration:.2f}s) =====")
        self._summary_df = self._generate_results_summary() 
        return self._summary_df


    def get_results_summary(self) -> Optional[pd.DataFrame]:
        if self._summary_df is not None:
            return self._summary_df
        elif self.results:
            self._summary_df = self._generate_results_summary()
            return self._summary_df
        else:
            print("No results collected yet. Run experiment first.")
            return None

    def _generate_results_summary(self) -> Optional[pd.DataFrame]:
        if not self.results:
            print("No results to summarize.")
            return None

        summary_data_list = []
        
        all_vars_encountered = set()
        for res in self.results:
            if res.get('error') is None: 
                if res.get('ghost_cox_relevance_df') is not None:
                    all_vars_encountered.update(res['ghost_cox_relevance_df'].index)
                if res.get('hrt_p_values') is not None:
                     all_vars_encountered.update(res['hrt_p_values'].keys())
        

        if not all_vars_encountered and self.results and self.results[0].get('n_features'):
            all_vars_list = [f'X{i+1}' for i in range(self.results[0]['n_features'])]
        elif all_vars_encountered:
            all_vars_list = sorted(list(all_vars_encountered))
        else: 
            all_vars_list = [] 


        for res in self.results:
            base_info = {
                'scenario_name': res['scenario_name'],
                'replicate_id': res['replicate_id'],
                'c_index': res.get('c_index', np.nan),
                'duration_sec': res.get('duration_sec', np.nan), 
                'ghost_estimator': res.get('ghost_estimator_type', 'N/A'),
                'survival_model': res.get('survival_model_type', 'N/A'),
                'error': res.get('error', None)
            }

            # Create one row in the summary per variable for this replicate
            for var_name in all_vars_list:
                row = base_info.copy()
                row['variable'] = var_name

                # GhostCox results
                if res.get('ghost_cox_relevance_df') is not None and isinstance(res['ghost_cox_relevance_df'], pd.DataFrame) and var_name in res['ghost_cox_relevance_df'].index:
                    row['relevance_score_gc'] = res['ghost_cox_relevance_df'].loc[var_name, 'RV_gh'] # Make sure 'RV_gh' is the final normalized column
                    row['relevance_rank_gc'] = res['ghost_cox_relevance_df'].loc[var_name, 'Rank']
                else:
                    row['relevance_score_gc'] = np.nan
                    row['relevance_rank_gc'] = np.nan
                
                # HRT results
                if res.get('hrt_p_values') is not None and isinstance(res['hrt_p_values'], dict):
                    row['p_value_hrt'] = res['hrt_p_values'].get(var_name, np.nan)
                else:
                    row['p_value_hrt'] = np.nan
                
                summary_data_list.append(row)
            
            # If a replicate failed entirely before variable processing, or if all_vars_list is empty
            if not all_vars_list and res.get('error') is not None:
                row = base_info.copy()
                row.update({'variable': 'N/A', 'relevance_score_gc': np.nan, 
                            'relevance_rank_gc': np.nan, 'p_value_hrt': np.nan})
                summary_data_list.append(row)


        if not summary_data_list:
            print("No data suitable for summary DataFrame construction.")
            return pd.DataFrame() # Return empty DataFrame

        summary_df = pd.DataFrame(summary_data_list)
        return summary_df

    # --- Plotting Methods ---


    def plot_p_value_comparison(self, variable: str, summary_df: Optional[pd.DataFrame] = None):
        """ Plots boxplots of HRT p-values for a specific variable across scenarios. """
        if summary_df is None:
            summary_df = self.get_results_summary()
            if summary_df is None or summary_df.empty:
                print("No summary data available for plotting p-values.")
                return

        var_df = summary_df[(summary_df['variable'] == variable) & summary_df['p_value_hrt'].notna()]
        if var_df.empty:
            print(f"No valid HRT p-value results found for variable '{variable}'.")
            return

        n_scenarios = len(var_df['scenario_name'].unique())
        plt.figure(figsize=(max(6, n_scenarios * 1.2), 5))
        sns.boxplot(data=var_df, x='scenario_name', y='p_value_hrt', hue='scenario_name', palette='viridis', legend=False)
        plt.axhline(0.05, ls='--', color='red', label='p=0.05') # Significance threshold line
        plt.xticks(rotation=45, ha='right')
        plt.title(f'HRT P-value Distribution for Variable: {variable}')
        plt.ylabel('P-value')
        plt.xlabel('Scenario')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()