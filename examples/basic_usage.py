"""Construct local signals for a small, chronologically separated panel."""

import numpy as np
from sklearn.linear_model import LinearRegression

from orthogonal_explanations import GaussianCOLA


def main():
    rng = np.random.default_rng(20261023)
    k = 6
    covariance = 0.6 * np.ones((k, k)) + 0.4 * np.eye(k)
    reference = rng.multivariate_normal(np.zeros(k), covariance, 512).astype(np.float32)
    future_inputs = rng.multivariate_normal(np.zeros(k), covariance, 24).astype(np.float32)
    weight = np.array([0.007, -0.005, 0.0035, 0.002, -0.0015, 0.001])
    historical_returns = reference @ weight + rng.normal(0, 0.005, len(reference))
    model = LinearRegression().fit(reference, historical_returns)
    explainer = GaussianCOLA(reference, model.predict,
                             linear_parameters=(model.coef_, model.intercept_))
    attributions, predictions = explainer.explain(future_inputs)
    error = np.max(np.abs(explainer.base + attributions.sum(axis=1) - predictions))
    assert error < 1e-12
    print(f"Explained {len(future_inputs)} future observations with {k} feature signals.")
    print(f"Maximum reconstruction error: {error:.3e}")
    # Keep these identities beside every attribution vector in a real panel.
    print({"asset": "asset_00", "information_date": "2020-12",
           "return_date": "2021-01", "learning_vintage": "2020-12",
           "prediction": float(predictions[0]), "baseline": explainer.base,
           "attributions": attributions[0].round(6).tolist()})


if __name__ == "__main__":
    main()
