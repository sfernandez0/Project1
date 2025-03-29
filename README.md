# Project 1 

# Lasso Homotopy Regression

## Team Members

* Arnau Fité Cluet
* Susana Fernández Cavia
* Antonio Ardura Carnicero

## Overview

This project provides a Python implementation of the Lasso (Least Absolute Shrinkage and Selection Operator) regression model using the Homotopy algorithm (similar to LARS - Least Angle Regression). The goal is to solve the linear regression problem with L1 regularization, which induces sparsity in the model coefficients and thereby performs automatic feature selection.

## Key Features

* **Lasso Regression (L1):** Implements L1-penalized linear regression from first principles.
* **Homotopy Algorithm:** Solves the regression problem by following the coefficient path as the regularization parameter `mu` varies.
* **Configurable:** Allows setting the regularization parameter (`mu`), numerical tolerance (`tol`), and maximum number of iterations (`max_iter`).
* **Familiar Interface:** Provides `.fit()` and `.predict()` methods similar to the Scikit-learn API.
* **Test Suite:** Includes unit and integration tests using `pytest` to verify model correctness and robustness.
* **Validation:** . This project includes a comprehensive set of tests implemented with PyTest to ensure that the Lasso Homotopy model behaves as expected under various scenarios. The tests cover different aspects of the model, from basic functionality to its performance on challenging datasets. It uses synthetic test data and direct comparison with scikit-learn’s Lasso model (in the file LassoComparisson-2.ipynb) to validate results. This will be explained further down in more detail. 

## Prerequisites

* Python (>= 3.7 recommended)
* `pip` (Python package installer)
* `git` (for cloning the repository)

## Installation

It is highly recommended to use a virtual environment to manage the project's dependencies.

1. **Clone the repository:**
    ```bash
    git clone https://github.com/sfernandez0/Project1.git  
    cd Project1
    ```

2. **Create and activate a virtual environment:**
    - On macOS/Linux:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    - On Windows:
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```

3. **Install dependencies:**
  Install the packages with the given requirements.txt file:
    ```bash
    pip install -r requirements.txt
    ```

## Questions

* What does the model you have implemented do and when should it be used?

  The model implements Lasso regression using the Homotopy Method. It is particularly useful for high-dimensional datasets where sparsity in the model coefficients is desired for feature selection and improved interpretability.
  
* How did you test your model to determine if it is working reasonably correctly?
  
  We developed a suite of tests using pytest, including:
    - Testing on small synthetic datasets with known true coefficients.
    - Testing on collinear datasets to ensure that the model produces sparse solutions.
    - Direct comparison of prediction accuracy and coefficient estimation with scikit-learn’s Lasso model (LassoComparisson.ipynb).
 
    To explain them in more detail:
    
    The first test, `test_small_dataset`, verifies that the model can be trained on a small dataset and produce valid predictions. In this test, a small CSV file is loaded and used to train the model with a moderate regularization parameter. The test then checks that the predictions have the correct shape and that all predicted values are finite, ensuring that no numerical issues (such as NaN or infinite values) occur during the computation.
    
    The `test_collinear_dataset` focuses on the model's ability to handle highly collinear data. It generates a synthetic dataset where each feature is a scaled version of a base feature, creating a scenario of extreme collinearity. The model is then trained on this data, and the test confirms that some of the coefficients are effectively zero, as expected when the model removes redundant features. This is a key characteristic of Lasso regression, where the L1 penalty should drive unnecessary coefficients to zero.
    
    Another test, `test_collinear_dataset2`, further examines the behavior of the model when applied to a real CSV dataset known to contain collinear features. In this test, the model is trained with a higher regularization parameter to enforce even stronger sparsity. The test verifies that the number of predictions matches the number of samples in the dataset and that the solution is sparse, meaning that fewer coefficients remain nonzero than the total number of features.
    
    The `test_higher_regularization` test is designed to demonstrate that increasing the regularization strength (i.e., the parameter `mu`) leads to a sparser solution. In this test, two models are trained on the same synthetic dataset – one with a very low value of `mu` and another with a high value. By comparing the number of nearly zero coefficients in both cases, the test shows that a higher regularization parameter indeed forces more coefficients towards zero, which is an expected behavior of the Lasso method.
    
    The `test_prediction_accuracy` test checks the accuracy of the model on a simple linear dataset. The dataset used in this test exhibits a clear linear relationship, and the model is trained with an extremely small regularization parameter to ensure that the fitted model closely follows the true linear relationship. The predictions from the model are then compared to the true values using a strict tolerance, verifying that the model’s outputs are accurate.

    Finally,the notebook `LassoComparisson-2.ipynb` directly compares the custom Lasso Homotopy model with scikit-learn's Lasso. It generates synthetic regression data with known sparse coefficients, trains both models on the same dataset, and evaluates prediction accuracy (using metrics like MSE) and coefficient sparsity. Visualizations, such as coefficient comparison plots and predicted vs. true value plots, help confirm that the custom implementation recovers the true sparse structure and performs similarly to scikit-learn's Lasso.

* What parameters have you exposed to users of your implementation in order to tune performance?

The following parameters can be tuned:
    - mu: The regularization parameter controlling the strength of the L1 penalty.
    - tol: The tolerance used to determine when a coefficient is effectively zero.
    - max_iter: The maximum number of iterations for the Homotopy algorithm.
These parameters allow users to adjust the level of sparsity, convergence precision, and computational cost.

* Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?
  The current implementation may face challenges with:
    - Extremely high-dimensional data or very high collinearity among features, which can lead to numerical instability.
    - Data that deviates significantly from the assumptions of linear regression (e.g., non-linearity or heteroscedasticity).
With additional time, improvements such as more robust numerical techniques (e.g., better regularized matrix inversion methods) and enhanced pre-processing steps could help mitigate these issues. However, some limitations may be inherent to the Lasso formulation in particularly challenging settings.

