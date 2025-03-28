import numpy as np
import logging
from typing import Tuple, Set, Dict, Optional, List, Union
# For predict, it is useful to use sklearn's check_is_fitted if available
try:
    from sklearn.utils.validation import check_is_fitted
except ImportError:
    # Define a dummy function if sklearn is not available
    def check_is_fitted(estimator, attributes=None):
        if attributes is None:
            attributes = ["coef_"]
        if not all(hasattr(estimator, attr) for attr in attributes):
            raise RuntimeError(f"This {type(estimator).__name__} instance is not fitted yet. Call 'fit' first.")

# --- Configuration and Constants ---
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  # INFO to see steps
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

EPSILON = 1e-10  # Default value, can be overwritten in __init__

# --- Class Definition ---

class LassoHomotopyModel():
    """
    LASSO model fitted iteratively using homotopy steps.

    Parameters
    ----------
    mu : float, default=0.1
        L1 regularization parameter (lambda).
    tol : float, default=1e-10
        Numerical tolerance for comparisons with zero.
    """
    def __init__(self, mu=0.1, tol=EPSILON):
        if mu < 0:
            raise ValueError("The regularization parameter mu must be nonnegative.")
        self.mu = mu
        self.tol = tol
        # Attributes that will be set after calling fit
        self.coef_ = None           # Final coefficients (theta)
        self.active_set_ = None     # Final active set (indices)
        self.signs_ = None          # Final signs
        self.K_ = None              # Final K matrix (Inverse of active Hessian)
        self.n_features_in_ = None  # Number of features seen in fit
        self.n_iter_ = 0            # Counter for total breakpoints found

    # --- Helper Functions (Converted to Private Methods) ---

    def _derive_state_from_theta(self, theta: np.ndarray) -> Tuple[Set[int], Dict[int, int]]:
        """Obtains the active set and signs from theta."""
        theta_flat = theta.flatten()
        # Use self.tol
        active_set = set(np.where(np.abs(theta_flat) > self.tol)[0])
        signs = {i: int(np.sign(theta_flat[i])) for i in active_set}
        return active_set, signs

    def _invupdateapp(self, A, x, y, r):
        """Update the inverse of a matrix by appending one row/column."""
        # Use self.tol instead of global EPSILON
        r_scalar = r.item() if isinstance(r, np.ndarray) and r.size == 1 else r
        if A.size == 0:
            if abs(r_scalar) < self.tol: return np.array([[np.inf]])  # Or handle differently
            return np.array([[1.0 / r_scalar]])
        yA = np.dot(y, A)
        dot_yA_x = np.dot(yA, x)
        den_scalar = r_scalar - dot_yA_x.item() if hasattr(dot_yA_x, 'item') else r_scalar - dot_yA_x

        if abs(den_scalar) < self.tol:
            logging.warning("invupdateapp: Denominator close to zero. Using pinv as fallback.")
            # Fallback: Rebuild original matrix and use pseudo-inverse
            # Note: This can be expensive and introduces approximation
            rows_A, cols_A = A.shape if A.ndim == 2 else (0, 0)
            # Reconstructing the matrix M = [[inv(A), x], [y, r]] is not straightforward.
            # What we need is inv([[H, C],[C.T, D]]) where H=X_old.T@X_old, C=X_old.T@x_new etc.
            # The safe fallback is to recalc K from scratch. We return None to signal failure here.
            logging.error("invupdateapp: Fallback with pinv not directly implemented here, signaling error.")
            return None  # Signal failure so the caller recalculates K

        q_scalar = 1.0 / den_scalar
        Ax = q_scalar * np.dot(A, x)
        Ax = Ax.reshape(-1, 1)
        yA = yA.reshape(1, -1)
        q_arr = np.array([[q_scalar]])
        try:
            block1 = np.hstack([A + np.dot(Ax, yA), -Ax])
            block2 = np.hstack([-yA * q_scalar, q_arr])
            return np.vstack([block1, block2])
        except ValueError as e:
            logging.error(f"invupdateapp: Error in vstack/hstack. Shapes: A={A.shape}, Ax={Ax.shape}, yA={yA.shape}. Error: {e}")
            raise

    def _invupdatered(self, A, c: int):
        """Update the inverse of a matrix by removing one row/column."""
        # Use self.tol
        n, m = A.shape
        if n <= 0 or n != m:  # If empty or not square
            return np.empty((0,0))
        if n == 1:  # Only one element left
            return np.empty((0,0))
        if not (0 <= c < n):
            logging.error(f"invupdatered: Index c={c} out of range for A with shape {A.shape}")
            raise IndexError("Index out of range in invupdatered")

        indn = np.arange(n)
        q_scalar = A[c, c]
        if abs(q_scalar) < self.tol:
            logging.warning("invupdatered: Pivot q close to zero. Using pinv as fallback.")
            # Fallback: Recalculate K from scratch. Return None.
            logging.error("invupdatered: Fallback with pinv not directly implemented here, signaling error.")
            return None  # Signal failure
        c1 = np.hstack((indn[:c], indn[c+1:]))
        Ax = A[c1, c].reshape(-1, 1)
        yA = A[c, c1].reshape(1, -1)
        A_sub = A[np.ix_(c1, c1)]
        try:
             return A_sub - (Ax @ yA) / q_scalar
        except Exception as e:
             logging.error(f"invupdatered: Error in final calculation: {e}")
             raise

    def _compute_regularization_step(self, X, y, theta_start, mu_start, mu_end) -> Tuple[np.ndarray, Set[int], Dict[int, int], np.ndarray]:
        """Computes the LASSO path by varying mu."""
        # --- Adapted from your code ---
        # Replace EPSILON with self.tol
        # Replace calls to _derive_state, invupdate*, etc., with self._...

        if mu_end < 0: raise ValueError("mu_end must be nonnegative.")
        n_samples, n_features = X.shape if X.ndim == 2 and X.size > 0 else (0, theta_start.shape[0])

        if n_samples == 0:  # No data case
             theta = np.zeros_like(theta_start)
             return theta, set(), {}, np.empty((0,0))

        # Initial state
        active_set, signs = self._derive_state_from_theta(theta_start)
        # nz must be sorted here for consistency if K is calculated with pinv
        nz = sorted(list(active_set))
        theta = theta_start.copy()
        current_mu = mu_start

        logging.info(f"RegStep: mu={mu_start:.4f} -> mu={mu_end:.4f}")

        # If there is no change in mu
        if abs(mu_start - mu_end) < self.tol:
            logging.debug("RegStep: mu unchanged.")
            K = np.empty((0,0))
            if nz:
                 try: K = np.linalg.pinv(X[:, nz].T @ X[:, nz])
                 except np.linalg.LinAlgError: K = np.eye(len(nz))  # Simple fallback
            return theta, set(nz), signs, K

        # Only implemented for descending path (mu_start > mu_end)
        if mu_start < mu_end:
            raise NotImplementedError("Ascending regularization path (fwd) not implemented.")

        # --- Descending Logic (bwd) ---
        max_lars_iter = 3 * n_features + 20  # More iterations just in case
        lars_iter_count = 0
        path_breakpoints = 0  # Count breakpoints in this step
        b = np.dot(X.T, y)
        G = np.dot(X.T, X)

        # Initial K for active set nz
        K = np.empty((0,0))
        if nz:
            X_nz_mat = X[:, nz]
            try: K = np.linalg.pinv(X_nz_mat.T @ X_nz_mat)
            except np.linalg.LinAlgError:
                logging.error("RegStep - Initial K pinv failed.")
                return theta, set(nz), signs, np.empty((0,0))  # Return initial state

        # v1 must correspond to the order of nz (which is sorted)
        v1 = np.array([signs[i] for i in nz]).reshape(-1, 1) if nz else np.empty((0,1))

        # Main LARS-like loop
        while current_mu > mu_end + self.tol and lars_iter_count < max_lars_iter:
            lars_iter_count += 1
            q = len(nz)
            logging.debug(f" RegStep - Iter {lars_iter_count}, mu={current_mu:.4f}, q={q}, nz={nz}")

            # --- Special handling if q=0 ---
            if q == 0:
                 residual = y  # theta is 0
                 correlations = np.abs(X.T @ residual)
                 if correlations.size == 0: break  # No features
                 mu_max_reactivate = np.max(correlations)
                 if mu_max_reactivate <= mu_end + self.tol:
                      logging.debug("RegStep - q=0, no feature will be activated.")
                      break  # Exit, theta remains zero

                 # Find where to reactivate
                 current_mu = mu_max_reactivate
                 if current_mu <= mu_end + self.tol: break  # Already below target

                 event_idx = np.argmax(correlations)
                 event_sign = np.sign((X[:, [event_idx]].T @ residual)[0, 0])
                 event_type = event_sign  # Use sign as type

                 logging.debug(f' RegStep - q=0: Reactivating at mu={current_mu:.4f}, F{event_idx}, Sign={event_sign}')
                 path_breakpoints += 1

                 # Update state for q=1
                 nz = [event_idx]
                 signs = {event_idx: event_sign}
                 v1 = np.array([[event_sign]])
                 try:
                      # Compute K for a single feature
                      x_new_sq = (X[:, [event_idx]].T @ X[:, [event_idx]])[0, 0]
                      if abs(x_new_sq) < self.tol: raise ValueError("Feature with zero norm.")
                      K = np.array([[1.0 / x_new_sq]])
                 except Exception as e:
                      logging.error(f"RegStep - Error computing K for reactivation: {e}")
                      break  # Critical failure
                 continue  # Go back to while loop with q=1

            # --- Main logic for q > 0 ---
            X_nz_mat = X[:, nz]; b_nz = b[nz]

            try:
                Kv1 = K @ v1
                Kb_nz = K @ b_nz
            except Exception as e:
                 logging.error(f"RegStep - Error in K@v1/K@b_nz: {e}. K:{K.shape}, v1:{v1.shape}, b_nz:{b_nz.shape}, nz:{nz}"); break

            # Find next breakpoint mu_next_loop
            mu_next_loop = mu_end
            event_type = -99  # Event type: 0=inactive, 1=active+, -1=active-
            event_idx = -1    # Global index of event feature
            event_idx_in_nz = -1  # Local index (within nz) if event_type=0

            # Event 1: Coefficient -> 0
            mu_event1 = -np.inf
            with np.errstate(divide='ignore', invalid='ignore'):
                 mu_0_candidates = Kb_nz / Kv1  # Shape (q, 1)
                 # Invalidate if Kv1 ~0 or if candidate mu is not between mu_end and current_mu
                 valid_mask_0 = (np.abs(Kv1) > self.tol) & \
                                (mu_0_candidates > mu_end + self.tol) & \
                                (mu_0_candidates < current_mu - self.tol)
                 mu_0_candidates[~valid_mask_0.flatten()] = -np.inf

            if np.any(mu_0_candidates > -np.inf):
                 idx_event1_local = np.argmax(mu_0_candidates)
                 mu_event1 = mu_0_candidates[idx_event1_local, 0]
                 if mu_event1 > mu_next_loop:  # Check if it is the closest to current_mu
                      mu_next_loop = mu_event1
                      event_type = 0
                      event_idx_in_nz = idx_event1_local
                      event_idx = nz[idx_event1_local]


            # Event 2: Inactive -> Active
            inactive_indices = sorted(list(set(range(n_features)) - set(nz)))
            mu_event2 = -np.inf  # Best mu found for activation
            idx_event2 = -1
            sign_event2 = 0

            if inactive_indices:
                z = inactive_indices; X_z = X[:, z]; b_z = b[z]
                M = G[np.ix_(z, nz)]
                try:
                    MKb_nz = M @ Kb_nz; MKv1 = M @ Kv1
                except Exception as e: logging.error(f"RegStep - Error M@...: {e}"); MKb_nz=0; MKv1=0  # Simplified

                # Candidates for activation +1
                with np.errstate(divide='ignore', invalid='ignore'):
                    den_1 = 1.0 - MKv1
                    mu_1_candidates = (b_z - MKb_nz) / den_1
                    valid_mask_1 = (np.abs(den_1) > self.tol) & \
                                   (mu_1_candidates > mu_end + self.tol) & \
                                   (mu_1_candidates < current_mu - self.tol)
                    mu_1_candidates[~valid_mask_1.flatten()] = -np.inf

                if np.any(mu_1_candidates > -np.inf):
                     idx_1_local_in_z = np.argmax(mu_1_candidates)
                     mu_cand_1 = mu_1_candidates[idx_1_local_in_z, 0]
                     if mu_cand_1 > mu_event2:  # Check if it is the best so far
                          mu_event2 = mu_cand_1
                          idx_event2 = z[idx_1_local_in_z]
                          sign_event2 = 1

                # Candidates for activation -1
                with np.errstate(divide='ignore', invalid='ignore'):
                    den_m1 = -1.0 - MKv1
                    mu_m1_candidates = (b_z - MKb_nz) / den_m1
                    valid_mask_m1 = (np.abs(den_m1) > self.tol) & \
                                    (mu_m1_candidates > mu_end + self.tol) & \
                                    (mu_m1_candidates < current_mu - self.tol)
                    mu_m1_candidates[~valid_mask_m1.flatten()] = -np.inf

                if np.any(mu_m1_candidates > -np.inf):
                     idx_m1_local_in_z = np.argmax(mu_m1_candidates)
                     mu_cand_m1 = mu_m1_candidates[idx_m1_local_in_z, 0]
                     if mu_cand_m1 > mu_event2:  # Check if it is the best so far
                          mu_event2 = mu_cand_m1
                          idx_event2 = z[idx_m1_local_in_z]
                          sign_event2 = -1

                # Compare best activation event with deactivation event
                if mu_event2 > mu_next_loop:
                     mu_next_loop = mu_event2
                     event_type = sign_event2
                     event_idx = idx_event2
                     event_idx_in_nz = -1  # Not applicable for activation


            # --- Determine actual breakpoint t_br and event ---
            t_br = 1.0
            candidates = [(t_cand1, 0, local_idx_cand1),  # (t, type=0, local index in nz)
                          (t_cand2_p, 1, local_idx_cand2_p),  # (t, type=1, local index in z)
                          (t_cand2_m, -1, local_idx_cand2_m)]  # (t, type=-1, local index in z)

            best_t = 1.0
            best_event = None

            for t_cand, type_cand, local_idx_cand in candidates:
                 if t_cand < best_t:
                      best_t = t_cand
                      best_event = (type_cand, local_idx_cand)

            if best_event is not None:
                 t_br = best_t
                 type_cand, local_idx_cand = best_event
                 if type_cand == 0:  # Deactivation
                      event_type = 0
                      event_idx_in_nz = local_idx_cand
                      if not (0 <= event_idx_in_nz < len(nz)): logging.error("HomoStep: Invalid index for deactivation."); event_type = -99; t_br=1.0
                      else: event_index = nz[event_idx_in_nz]
                 elif type_cand == 1:  # Activation +1
                      event_type = 2
                      sign_event2 = 1
                      if not (0 <= local_idx_cand < len(inactive_indices)): logging.error("HomoStep: Invalid index for activation +1."); event_type = -99; t_br=1.0
                      else: event_index = inactive_indices[local_idx_cand]
                 elif type_cand == -1:  # Activation -1
                      event_type = 2
                      sign_event2 = -1
                      if not (0 <= local_idx_cand < len(inactive_indices)): logging.error("HomoStep: Invalid index for activation -1."); event_type = -99; t_br=1.0
                      else: event_index = inactive_indices[local_idx_cand]
            else:
                 t_br = 1.0
                 event_type = -99

            # --- Update t and state ---
            current_t = t_br

            if current_t < 1.0 - self.tol:  # If an event occurred before t=1
                path_breakpoints += 1
                recalculate_K_needed = False

                if event_type == 0:  # Deactivate
                    logging.debug(f' HomoStep - T: t={current_t:.4f}, F{event_index} inactive')
                    if event_index not in nz or event_idx_in_nz < 0: logging.error(f"HomoStep Inact Err {event_index}/{event_idx_in_nz} from {nz}"); break
                    if not (0 <= event_idx_in_nz < len(nz)): logging.error(f"HomoStep Inact Idx Err {event_idx_in_nz}"); break

                    K_new = self._invupdatered(K, event_idx_in_nz) if K is not None else None
                    if K_new is None:  # Rank-1 update failed
                          recalculate_K_needed = True; K = None
                    else:
                          K = K_new

                    # Update nz and signs
                    try:
                         original_pos = nz.index(event_index)
                         nz.pop(original_pos)
                    except ValueError: logging.error(f"HomoStep: Error removing {event_index} from {nz}"); break
                    if event_index in signs: del signs[event_index]

                elif event_type == 2:  # Activate
                    logging.debug(f' HomoStep - T: t={current_t:.4f}, F{event_index} active ({sign_event2})')
                    if event_index in nz: logging.error(f"HomoStep Act Err {event_index} in {nz}"); break

                    X_nz_old = X[:, nz] if nz else np.empty((n_total_samples, 0))
                    X_new_col = X[:, [event_index]]
                    K_new = self._invupdateapp(K, X_nz_old.T @ X_new_col, X_new_col.T @ X_nz_old,
                                               X_new_col.T @ X_new_col) if K is not None else None
                    if K_new is None:  # Rank-1 update failed
                          recalculate_K_needed = True; K = None
                    else:
                          K = K_new

                    # Update nz and signs
                    nz.append(event_index)  # Append AFTER invupdateapp
                    signs[event_index] = sign_event2

                else:
                    logging.warning(f"HomoStep - Unknown event type {event_type} at t={current_t}. Stopping."); current_t = 1.0; break

                # Recalculate K if rank-1 update failed
                if recalculate_K_needed:
                     logging.warning("HomoStep: Recalculating K with pinv due to rank-1 update failure.")
                     current_nz_sorted = sorted(list(nz))  # Sort for consistent pinv
                     if not current_nz_sorted: K = np.empty((0,0))
                     else:
                          try: K = np.linalg.pinv(X[:, current_nz_sorted].T @ X[:, current_nz_sorted])
                          except np.linalg.LinAlgError: logging.error("HomoStep: pinv fallback failed."); K=None; break
                          nz = current_nz_sorted
                     logging.debug(f"HomoStep: Recalculated K, shape={K.shape if K is not None else 'None'}, nz={nz}")

                # Update v1 for the next iteration (respecting order of nz)
                v1 = np.array([signs[i] for i in nz]).reshape(-1, 1) if nz else np.empty((0,1))

            else:  # t reached 1
                logging.debug(f"HomoStep - Reached t=1.0")
                current_t = 1.0

        # --- End of Homotopy Loop ---
        if homotopy_iter_count >= max_homotopy_iter:
            logging.warning(f"HomoStep - Max iter ({max_homotopy_iter}) reached.")
        self.n_iter_ += path_breakpoints

        # --- Compute final theta at t = 1 ---
        theta.fill(0.0)
        K_final = K  # K at end of loop (could be None if failed)
        final_nz_calc = list(nz)  # nz at end of loop
        final_signs_calc = signs.copy()

        if final_nz_calc:
            logging.debug("HomoStep - Recalculating final K with pinv for robustness.")
            final_nz_sorted = sorted(final_nz_calc)
            try:
                K_final = np.linalg.pinv(X[:, final_nz_sorted].T @ X[:, final_nz_sorted])
                final_nz_calc = final_nz_sorted
                final_signs_calc = {i: signs[i] for i in final_nz_calc}
            except np.linalg.LinAlgError:
                logging.error("HomoStep Final - pinv failed.")
                K_final = None
                final_nz_calc = []
                final_signs_calc = {}

            if K_final is not None and K_final.size > 0:
                 b_nz = b[final_nz_calc]
                 v1 = np.array([final_signs_calc[i] for i in final_nz_calc]).reshape(-1, 1)
                 if K_final.shape[0] == len(final_nz_calc) and v1.shape[0] == len(final_nz_calc):
                      try:
                           theta_nz_final = K_final @ (b_nz - mu_fixed * v1)
                           theta[final_nz_calc, 0] = theta_nz_final.flatten()
                      except Exception as e:
                           logging.error(f"HomoStep Final - Error computing theta: {e}")
                           theta.fill(np.nan)
                 else:
                      logging.warning(f"HomoStep Final - K_final {K_final.shape} or v1 {v1.shape} inconsistent with nz ({len(final_nz_calc)}). Theta set to zero.")
                      final_nz_calc = []; final_signs_calc = {}
            # else: K_final failed, theta remains zero
        # else: nz was empty, theta remains zero

        theta[np.abs(theta) < self.tol] = 0.0
        active_set_final, signs_final = self._derive_state_from_theta(theta)
        final_nz_list = sorted(list(active_set_final))  # For logging
        logging.info(f"HomoStep: Finished at t={current_t:.4f}. Actives: {final_nz_list}")

        # Return state consistent with the computed final theta
        return theta, active_set_final, signs_final, K_final

    def fit(self, X, y):
        """
        Fits the LASSO Homotopy model iteratively to the data X, y.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,) or (n_samples, 1)
            Target values.

        Returns
        -------
        self : object
            The fitted model instance.
        """
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        self.n_iter_ = 0  # Reset breakpoint counter

        # Ensure y is a column vector (n_samples, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # --- Initialization (First point i=0) ---
        logging.info("--- Fit: Initialization (Point 0) ---")
        X0 = X[0, :].reshape(1, n_features)
        y0 = y[0, :].reshape(1, 1)

        # Compute mu_max for the first point
        mu_max_0 = np.max(np.abs(X0.T @ y0)) if X0.size > 0 else 0

        if self.mu >= mu_max_0 - self.tol:
             # Initial solution is zero
             theta_current = np.zeros((n_features, 1))
             active_set_current = set()
             signs_current = {}
             K_current = np.empty((0,0))
             logging.info(f"Fit Init: mu={self.mu:.4f} >= mu_max_0={mu_max_0:.4f}. Initial theta is zero.")
        else:
             # Compute initial state using regularization step
             try:
                 logging.debug(f"Fit Init: Computing RegStep from mu={mu_max_0:.4f} to {self.mu:.4f}")
                 theta_current, active_set_current, signs_current, K_current = self._compute_regularization_step(
                     X=X0,
                     y=y0,
                     theta_start=np.zeros((n_features, 1)),
                     mu_start=mu_max_0,  # Start from mu_max
                     mu_end=self.mu      # Go to model mu
                 )
                 if np.any(np.isnan(theta_current)): raise ValueError("NaN in initial theta.")
             except Exception as e:
                 logging.error(f"Fit: RegularizationStep failed during initialization: {e}", exc_info=True)
                 theta_current = np.zeros((n_features, 1))
                 active_set_current = set(); signs_current = {}; K_current = np.empty((0,0))
                 logging.warning("Fit: Continuing from theta=0 after initialization failure.")

        logging.info(f"Fit Init: Theta={theta_current.T.round(4)}, nz={sorted(list(active_set_current))}")

        # --- Iterative Updates (Points i=1 to n-1) ---
        for i in range(1, n_samples):
            logging.info(f"--- Fit: Processing Point {i} ---")
            X_n = X[:i, :]
            y_n = y[:i, :]  # y is already (i, 1)
            x_new = X[i, :]
            y_new = y[i, 0]  # Scalar y for the new point

            # State from the previous iteration
            theta_n = theta_current.copy()
            K_n = K_current  # K from iteration i-1

            # Since mu is constant, Step 1 (Regularization) is trivial
            theta_intermediate = theta_n
            active_set_intermediate = active_set_current
            signs_intermediate = signs_current
            K_intermediate = K_n  # K corresponds to X_n, theta_n

            # --- Step 2: Homotopy Path (Add Point i) ---
            try:
                logging.debug(f"Fit Pt {i}: Computing HomotopyStep...")
                theta_current, active_set_current, signs_current, K_current = self._compute_homotopy_step(
                    X_n=X_n,
                    y_n=y_n,
                    theta_start=theta_intermediate,
                    K_start=K_intermediate,  # K for X_n[nz]
                    x_new=x_new,
                    y_new=y_new,
                    mu_fixed=self.mu  # mu remains constant
                )
                if np.any(np.isnan(theta_current)):
                     raise ValueError("NaN encountered in theta during HomotopyStep.")

            except Exception as e:
                logging.error(f"Fit: HomotopyStep failed for point {i}: {e}", exc_info=True)
                self.coef_ = theta_n
                self.active_set_ = self._derive_state_from_theta(theta_n)[0]
                self.signs_ = self._derive_state_from_theta(theta_n)[1]
                logging.warning("Fit: Terminating early due to error.")
                return self

            logging.info(f"Fit Pt {i}: Theta={theta_current.T.round(4)}, nz={sorted(list(active_set_current))}")

        # --- Save Final Results ---
        self.coef_ = theta_current
        self.active_set_ = active_set_current
        self.signs_ = signs_current
        self.K_ = K_current  # Save the last computed K (may be None if failed)
        logging.info("--- Fit Completed ---")
        logging.info(f"Total breakpoints found: {self.n_iter_}")

        return self

    # --- Predict Method ---
    def predict(self, X):
        """
        Predicts values for X using the fitted model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data for prediction.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        check_is_fitted(self, ['coef_', 'n_features_in_'])
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but the model was fitted with {self.n_features_in_}")

        y_pred = X @ self.coef_
        return y_pred.flatten()

# --- Results Class (Optiona) ---
class LassoHomotopyResults():
    """Stores the results of fitting a LassoHomotopyModel."""
    def __init__(self, coef, active_set, signs, K, n_features_in):
         self.coef_ = coef
         self.active_set_ = active_set
         self.signs_ = signs
         self.K_ = K
         self.n_features_in_ = n_features_in

    def predict(self, X):
         """Predict using the stored results."""
         if not hasattr(self, 'coef_') or not hasattr(self, 'n_features_in_'):
              raise RuntimeError("Results object not properly initialized.")
         if X.shape[1] != self.n_features_in_:
              raise ValueError(f"X has {X.shape[1]} features, but results are from {self.n_features_in_}")

         y_pred = X @ self.coef_
         return y_pred.flatten()
