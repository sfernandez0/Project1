# Project 1 
Lasso Homotopy Regression

Overview

This project implements LASSO (Least Absolute Shrinkage and Selection Operator) regression using a Homotopy method. In this approach, the model is updated iteratively when new data points are added. The LASSO adds an L1 penalty to ordinary least squares regression to enforce sparsity, effectively performing feature selection by shrinking irrelevant coefficients to zero. This implementation is particularly useful in online settings and when handling high-dimensional data.

Group Members

Susana Fernandez Cavia
Arnau Fité Cluet
Antonio Ardura
Features

Iterative Model Updating:
The model is updated point by point using a two-step process:
Regularization Step: Updates the solution by varying the regularization parameter.
Homotopy Step: Computes the homotopy path (from t=0 to t=1) to incorporate a new data point.
Active Set Management:
Maintains and updates the active set and signs of coefficients during the homotopy path.
Robust Matrix Updates:
Uses rank-1 updates for the inverse Hessian (K) of the active variables.
Predict Function:
Once the model is fitted, you can predict new data points.
Installation & Setup

Follow these steps to set up and run the code:

1. Clone the Repository
bash
Copy
git clone https://github.com/YourUsername/YourRepository.git
cd YourRepository
2. Create & Activate Virtual Environment
Windows:

bash
Copy
python -m venv venv
.\venv\Scripts\activate
macOS/Linux:

bash
Copy
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
Install the required packages via pip:

bash
Copy
pip install -r requirements.txt
The dependencies include (but are not limited to):

numpy
scikit-learn


* What does the model you have implemented do and when should it be used?
* How did you test your model to determine if it is working reasonably correctly?
* What parameters have you exposed to users of your implementation in order to tune performance? 
* Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?
