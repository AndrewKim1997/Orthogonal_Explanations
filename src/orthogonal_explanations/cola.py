"""Ordered conditional increments under a Ledoit-Wolf Gaussian reference.

The natural-order calculation follows Experiment 3. Exact linear expectations
are also used in the case study. Paper experiments retain their own original
implementations; this module exposes a small interface for reuse.
"""

import numpy as np
from sklearn.covariance import LedoitWolf


def _order(order, k):
    values = tuple(range(k)) if order is None else tuple(order)
    if (len(values) != k or any(isinstance(j, (bool, np.bool_)) or
                              not isinstance(j, (int, np.integer)) for j in values)
            or sorted(values) != list(range(k))):
        raise ValueError("order must be a permutation of the feature indices.")
    return np.asarray(values, dtype=int)


def ordered_cola(values, k, order=None):
    """Apply ordered increments to a complete bitmask-indexed coalition table.

    ``values[:, mask]`` contains the conditional value of coalition ``mask``.
    Returned columns retain their original feature identities for every order.
    This is the coalition-table calculation in Experiment 1.
    """
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("k must be a positive integer.")
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 1 << k or not np.isfinite(values).all():
        raise ValueError("values must be a finite matrix with 2**k columns.")
    indices = _order(order, k)
    attr = np.zeros((len(values), k))
    prefix = 0
    for j in indices:
        nxt = prefix | (1 << int(j))
        attr[:, j] = values[:, nxt] - values[:, prefix]
        prefix = nxt
    return attr


class GaussianCOLA:
    """Explain a scalar predictor through ordered Gaussian expectations.

    Parameters
    ----------
    x_reference : array of shape (n_reference, n_features)
        Pretest inputs in the predictor's transformed feature coordinates.
    predict : callable
        Scalar predictions in return units, one per input row.
    linear_parameters : (weights, intercept), optional
        Exact coefficients of ``predict`` in those same coordinates and units.
        Supplying these selects analytical Gaussian expectations.
    order : permutation, optional
        Prespecified feature order; defaults to input-column order.
    draws, baseline_draws, batch_size : int
        Antithetic conditional draws, baseline draws, and row batching.
    seed : int
        Baseline integration seed. ``explain`` takes a separate conditional seed.

    ``base`` stores the fitted reference baseline. With exact expectations,
    increments are orthogonal under that reference distribution. Finite draws,
    empirical monthly cross-sections, and distribution shifts affect the observed
    correlations. The sum telescopes to ``prediction - base`` in every case.
    """

    def __init__(self, x_reference, predict, *, linear_parameters=None, order=None,
                 draws=16, baseline_draws=1024, batch_size=512, seed=0):
        x_reference = np.asarray(x_reference)
        if (x_reference.ndim != 2 or len(x_reference) < 2 or
                x_reference.shape[1] < 1 or not np.isfinite(x_reference).all()):
            raise ValueError("x_reference must be a finite matrix with at least two rows.")
        if not callable(predict):
            raise TypeError("predict must be callable.")
        for name, value in (("draws", draws), ("baseline_draws", baseline_draws),
                            ("batch_size", batch_size)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if draws < 2 or draws % 2:
            raise ValueError("draws must be an even integer >= 2.")
        self.k = x_reference.shape[1]
        self.order = _order(order, self.k)
        self.draws, self.batch_size = int(draws), int(batch_size)
        self._predict_fn = predict
        fit = LedoitWolf().fit(x_reference)
        self.mean = np.asarray(fit.location_, float)
        self.cov = np.asarray(fit.covariance_, float)
        self.prefixes = []
        for size in range(1, self.k):
            obs, miss = self.order[:size], self.order[size:]
            cross = self.cov[np.ix_(obs, miss)]
            gain = np.linalg.solve(self.cov[np.ix_(obs, obs)], cross)
            conditional_cov = self.cov[np.ix_(miss, miss)] - cross.T @ gain
            eigenvalues, eigenvectors = np.linalg.eigh((conditional_cov + conditional_cov.T) / 2)
            root = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0)))
            self.prefixes.append((obs, miss, gain, root))
        self.weight = self.intercept = None
        if linear_parameters is not None:
            weight, intercept = linear_parameters
            self.weight = np.asarray(weight, dtype=float).copy()
            self.intercept = float(intercept)
            if (self.weight.shape != (self.k,) or not np.isfinite(self.weight).all()
                    or not np.isfinite(self.intercept)):
                raise ValueError("linear_parameters must contain finite weights and a scalar intercept.")
            self.base = float(self.intercept + self.mean @ self.weight)
        else:
            rng = np.random.default_rng(seed)
            background = rng.multivariate_normal(self.mean, self.cov, int(baseline_draws))
            self.base = float(self._predict(background).mean())

    def _predict(self, x):
        result = np.asarray(self._predict_fn(x), dtype=float)
        if result.shape != (len(x),) or not np.isfinite(result).all():
            raise ValueError("predict must return one finite scalar per input row.")
        return result

    def explain(self, x, seed=0):
        """Return ``(attributions, prediction)`` with original feature columns.

        Inputs use float32, matching Experiment 3; outputs use float64. Fit the
        reference and predictor on historical observations before explaining a
        forecast window. Reuse this fitted object for that learning vintage.
        """
        x = np.asarray(x, np.float32)
        if x.ndim != 2 or x.shape[1] != self.k or not np.isfinite(x).all():
            raise ValueError("x must be finite with the reference feature count.")
        attr = np.empty((len(x), self.k), float)
        previous = np.full(len(x), self.base)
        rng = np.random.default_rng(seed)
        for obs, miss, gain, root in self.prefixes:
            center = self.mean[miss] + (x[:, obs] - self.mean[obs]) @ gain
            if self.weight is not None:
                current = (self.intercept + x[:, obs] @ self.weight[obs]
                           + center @ self.weight[miss])
            else:
                z = rng.standard_normal((self.draws // 2, len(miss)))
                z = np.concatenate([z, -z], axis=0) @ root.T
                current = np.empty(len(x))
                for start in range(0, len(x), self.batch_size):
                    stop = min(len(x), start + self.batch_size)
                    filled = np.broadcast_to(x[start:stop, None, :],
                                             (stop - start, self.draws, self.k)).copy()
                    filled[:, :, miss] = center[start:stop, None, :] + z[None, :, :]
                    current[start:stop] = self._predict(filled.reshape(-1, self.k)).reshape(-1, self.draws).mean(axis=1)
            attr[:, obs[-1]] = current - previous
            previous = current
        prediction = self._predict(x)
        attr[:, self.order[-1]] = prediction - previous
        return attr, prediction
