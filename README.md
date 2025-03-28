# Project 1 

# Lasso Homotopy Regression

## Overview

This project provides a Python implementation of the Lasso (Least Absolute Shrinkage and Selection Operator) regression model using the Homotopy algorithm (similar to LARS - Least Angle Regression). The goal is to solve the linear regression problem with L1 regularization, which induces sparsity in the model coefficients and thereby performs automatic feature selection.

## Key Features

* **Lasso Regression (L1):** Implements L1-penalized linear regression from first principles.
* **Homotopy Algorithm:** Solves the regression problem by following the coefficient path as the regularization parameter `mu` varies.
* **Configurable:** Allows setting the regularization parameter (`mu`), numerical tolerance (`tol`), and maximum number of iterations (`max_iter`).
* **Familiar Interface:** Provides `.fit()` and `.predict()` methods similar to the Scikit-learn API.
* **Test Suite:** Includes unit and integration tests using `pytest` to verify model correctness and robustness.
* **Validation:** Uses synthetic test data and direct comparison with scikit-learn’s Lasso model to validate results.

## Prerequisites

* Python (>= 3.7 recommended)
* `pip` (Python package installer)
* `git` (for cloning the repository)

## Installation

It is highly recommended to use a virtual environment to manage the project's dependencies.

1. **Clone the repository:**
    ```bash
    git clone <your-repository-url>  # Replace with the actual URL
    cd <repository-directory-name>
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
    Create a file named `requirements.txt` with the following content:
    ```txt
    numpy>=1.18.0
    scikit-learn>=0.24.0
    pytest>=6.0.0
    ```
    Then, install the packages:
    ```bash
    pip install -r requirements.txt
    ```

* What does the model you have implemented do and when should it be used?

  The implemented model is an iterative LASSO regression model that uses a homotopy method to update its solution as new data points are added. It maintains an active set of features along with their signs, and it updates the inverse Hessian (matrix K) incrementally using rank-1 updates. This model is particularly useful in online learning scenarios where data arrives sequentially and you need to update the regression solution in real time. It is also suitable for high-dimensional datasets where only a few features are relevant (i.e., when a sparse solution is expected).
* How did you test your model to determine if it is working reasonably correctly?
  
* What parameters have you exposed to users of your implementation in order to tune performance?

  The model exposes the following key parameters for performance tuning:
    - mu (Regularization Parameter): This parameter (often denoted as lambda) controls the strength of the L1 penalty. Adjusting mu allows users to control the sparsity level of the solution.
    - tol (Numerical Tolerance): This parameter sets the threshold for comparisons with zero, ensuring numerical stability during the iterative updates. It affects how the model determines if a coefficient is active or should be set to zero.
These parameters enable users to fine-tune the balance between sparsity and fitting accuracy, as well as the numerical stability of the solution.
* Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?
