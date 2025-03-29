import csv
import os
import warnings

import numpy as np
import pytest
from model.LassoHomotopy import LassoHomotopyModel

def load_csv_data(filename, target_col='target'):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, filename)

    data = []
    with open(csv_path, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)

    if not data:
        raise ValueError("Loaded CSV data is empty.")

    # Check exact column names
    print("Column names:", data[0].keys())  # For debugging

    X = np.array([[float(v) for k, v in datum.items() if k.startswith('X') or k.startswith('x')] for datum in data])
    y = np.array([float(datum[target_col]) for datum in data])

    if X.size == 0 or y.size == 0:
        raise ValueError("X or y is empty after loading CSV data.")

    return X, y


def test_small_dataset():
    X, y = load_csv_data("small_test.csv", target_col='y')
    model = LassoHomotopyModel(mu=0.1)
    results = model.fit(X, y)
    preds = results.predict(X)

    assert preds.shape == y.shape
    assert np.all(np.isfinite(preds))

def test_collinear_dataset():
    np.random.seed(0)
    X_base = np.random.randn(100, 1)
    X_collinear = np.hstack([X_base, 2 * X_base, 3 * X_base])  # clearly collinear
    y = X_base[:, 0] + np.random.randn(100) * 0.01

    model = LassoHomotopyModel(mu=0.1)
    results = model.fit(X_collinear, y)
    sparsity = np.sum(np.abs(results.coefficients) < 1e-6)

    assert sparsity > 0, "Model did not produce sparse solution on collinear data"

def test_collinear_dataset2():
    
    X, y = load_csv_data("collinear_data.csv")
    model = LassoHomotopyModel(mu=10)
    results = model.fit(X, y)
    preds = results.predict(X)
    
    # Check that the shape of the predictions is correct
    assert preds.shape[0] == X.shape[0], "The shape of the predictions is not correct"
    
    # It is expected that with collinear data the solution is sparse.
    nonzero_coef = np.sum(np.abs(model.coef_) > model.tol)
    print(model.coef_)
    assert nonzero_coef < X.shape[1], "The model did not produce a sparse solution on collinear data"

def test_higher_regularization():
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(100) * 0.01

    model_low_mu = LassoHomotopyModel(mu=0.0001)
    results_low_mu = model_low_mu.fit(X, y)
    sparsity_low = np.sum(np.abs(results_low_mu.coefficients) < 1e-6)

    model_high_mu = LassoHomotopyModel(mu=10.0)  # Even more explicit
    results_high_mu = model_high_mu.fit(X, y)
    sparsity_high = np.sum(np.abs(results_high_mu.coefficients) < 1e-6)

    assert sparsity_high > sparsity_low, "Higher mu didn't increase sparsity"

def test_prediction_accuracy():
    X = np.array([[1, 2], [2, 4], [3, 6], [4, 8]])
    y = np.array([3, 6, 9, 12])

    model = LassoHomotopyModel(mu=0.000001)  # very small for high accuracy
    results = model.fit(X, y)
    preds = results.predict(X)

    np.testing.assert_allclose(preds, y, atol=0.1, err_msg="Predictions not accurate enough")


if __name__ == "__main__":
    pytest.main()
