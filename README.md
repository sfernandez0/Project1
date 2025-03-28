# Project 1 

# Lasso Homotopy Regression

## Overview

This project implements **LASSO** (Least Absolute Shrinkage and Selection Operator) regression using a **Homotopy method**. In this approach, the model is updated iteratively when new data points are added. The LASSO adds an L1 penalty to ordinary least squares regression to enforce sparsity, effectively performing feature selection by shrinking irrelevant coefficients to zero. This implementation is particularly useful in online settings and when handling high-dimensional data.

## Group Members

- Susana Fernandez Cavia  
- Arnau Fité Cluet  
- Antonio Ardura

## Features

- **Iterative Model Updating:**  
  The model is updated point by point using a two-step process:
  - **Regularization Step:** Updates the solution by varying the regularization parameter.
  - **Homotopy Step:** Computes the homotopy path (from t=0 to t=1) to incorporate a new data point.
- **Active Set Management:**  
  Maintains and updates the active set and signs of coefficients during the homotopy path.
- **Robust Matrix Updates:**  
  Uses rank-1 updates for the inverse Hessian (K) of the active variables.
- **Predict Function:**  
  Once the model is fitted, you can predict new data points.

## Installation & Setup

Follow these steps to set up and run the code:

### 1. Clone the Repository

```bash
git clone https://github.com/YourUsername/YourRepository.git
cd YourRepository
```

### 2. Create & Activate Virtual Environment

**Windows:**

```bash
python -m venv venv
.\venv\Scripts\activate
```
**macOS/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
The dependencies include (but are not limited to):

numpy
scikit-learn

* What does the model you have implemented do and when should it be used?

  The implemented model is an iterative LASSO regression model that uses a homotopy method to update its solution as new data points are added. It maintains an active set of features along with their signs, and it updates the inverse Hessian (matrix K) incrementally using rank-1 updates. This model is particularly useful in online learning scenarios where data arrives sequentially and you need to update the regression solution in real time. It is also suitable for high-dimensional datasets where only a few features are relevant (i.e., when a sparse solution is expected).
* How did you test your model to determine if it is working reasonably correctly?
  
* What parameters have you exposed to users of your implementation in order to tune performance?

  The model exposes the following key parameters for performance tuning:
    - mu (Regularization Parameter): This parameter (often denoted as lambda) controls the strength of the L1 penalty. Adjusting mu allows users to control the sparsity level of the solution.
    - tol (Numerical Tolerance): This parameter sets the threshold for comparisons with zero, ensuring numerical stability during the iterative updates. It affects how the model determines if a coefficient is active or should be set to zero.
These parameters enable users to fine-tune the balance between sparsity and fitting accuracy, as well as the numerical stability of the solution.
* Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?
