import numpy as np


class LassoHomotopyModel:
    def __init__(self, mu=0.1, tol=1e-6, max_iter=1000):
        self.mu = mu
        self.tol = tol
        self.max_iter = max_iter
        self.coef_ = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        theta = np.zeros(n_features)
        residual = y - X @ theta
        active_set = []
        signs = []

        mu_current = np.max(np.abs(X.T @ residual))
        mu_final = self.mu
        iteration = 0
        epsilon = 1e-4  # small decrement to avoid stalling

        while mu_current > mu_final and iteration < self.max_iter:
            iteration += 1
            correlations = X.T @ residual
            abs_corr = np.abs(correlations)

            # Feature to add to the active set
            j_max = np.argmax(abs_corr)
            if j_max not in active_set:
                active_set.append(j_max)
                signs.append(np.sign(correlations[j_max]))

            X_A = X[:, active_set]
            s = np.array(signs)

            try:
                G_inv = np.linalg.inv(X_A.T @ X_A)
            except np.linalg.LinAlgError:
                G_inv = np.linalg.pinv(X_A.T @ X_A)

            theta_A = G_inv @ (X_A.T @ y - mu_current * s)
            theta[:] = 0
            theta[active_set] = theta_A

            residual = y - X @ theta

            # Check for coefficients becoming zero
            to_remove = [i for i, idx in enumerate(active_set) if abs(theta[idx]) < self.tol]
            for i in reversed(to_remove):
                del active_set[i]
                del signs[i]

            # Update mu
            inactive_set = [j for j in range(n_features) if j not in active_set]
            if inactive_set:
                mu_next = np.max(np.abs(correlations[inactive_set]))
                mu_current = max(mu_final, mu_next - epsilon)
            else:
                mu_current = mu_final

        self.coef_ = theta
        return LassoHomotopyResults(theta)


class LassoHomotopyResults:
    def __init__(self, coefficients):
        self.coefficients = coefficients

    def predict(self, X):
        return X @ self.coefficients
