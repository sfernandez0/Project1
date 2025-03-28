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

  The model implements Lasso regression using the Homotopy Method. It is particularly useful for high-dimensional datasets where sparsity in the model coefficients is desired for feature selection and improved interpretability.
  
* How did you test your model to determine if it is working reasonably correctly?
  We developed a suite of tests using pytest, including:
    - Testing on small synthetic datasets with known true coefficients.
    - Testing on collinear datasets to ensure that the model produces sparse solutions.
    - Direct comparison of prediction accuracy and coefficient estimation with scikit-learn’s Lasso model (LassoComparisson.ipynb).
  
* What parameters have you exposed to users of your implementation in order to tune performance?

The following parameters can be tuned:
    - mu: The regularization parameter controlling the strength of the L1 penalty.
    - tol: The tolerance used to determine when a coefficient is effectively zero.
    - max_iter: The maximum number of iterations for the Homotopy algorithm.
These parameters allow users to adjust the level of sparsity, convergence precision, and computational cost.

* Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?
