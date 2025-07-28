# GhostCox# GhostCox: Enhancing Interpretability of Non-linear Proportional Hazard Models

This repository contains the official implementation for the paper **"Enhancing the Interpretability of Non-linear Proportional Hazard Models Introducing Ghost Variables"**

GhostCox is a novel method to enhance interpretability for non-linear survival models by quantifying the unique, conditional relevance of each covariate to the learned risk function.

---

## Key Features

-   **Robust to Correlations:** Accurately assesses variable importance even in the presence of highly correlated features.
-   **Computationally Efficient:** Orders of magnitude faster than comparable methods like the Holdout Randomization Test (HRT), making it practical for real-world datasets.
-   **Rich Interpretations:** Provides both individual variable relevance scores ($RV_{gh}$) and insights into joint variable effects through the Relevance Matrix (V).
-   **Model-Agnostic Core:** While designed for survival models, the core logic can be adapted to interpret any complex predictive model.

---

## Methodology Overview

GhostCox adapts the **Ghost Variable** framework (Delicado and Peña, 2023) to the proportional hazards setting. The core idea is to measure a variable's importance by observing how much the model's predicted risk score, $\hat{f}(x)$, changes when the variable is replaced by its "ghost": its conditional expectation given all other variables, $\hat{\mathbb{E}}(X_j | X_{-j})$.

This is achieved in a two-stage process:
1.  **Model Fitting:** A non-linear survival model (e.g., Random Survival Forest) is fitted on a training set to get the risk function $\hat{f}(x)$.
2.  **Interpretation:** On an independent test set, the Ghost Variable analysis is performed to calculate relevance scores and the Relevance Matrix.

---