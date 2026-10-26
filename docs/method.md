# COLA and the reusable API

## Ordered explanation coordinates

Fix a fitted scalar predictor $f$, a reference distribution $P$, and a prespecified feature order $\pi$. Let $S_j$ contain the first $j$ features in that order and define

$$
v_j(x)=E_P[f(X)\mid X_{S_j}=x_{S_j}],\qquad
v_0(x)=b=E_P[f(X)],\qquad v_K(x)=f(x).
$$

The coordinate attached to feature $\pi_j$ is

$$
a_{\pi_j}(x)=v_j(x)-v_{j-1}(x).
$$

The increments telescope to $f(x)-b$. For square-integrable $f$ and exact conditional expectations, they have conditional mean zero given earlier features and are mutually orthogonal under $P$. Their labels describe the information added after preceding features. Changing the order changes that allocation.

The implementation estimates a Gaussian reference with Ledoit–Wolf shrinkage. Linear predictors admit analytical conditional expectations. General predictors use shared antithetic Gaussian draws for each prefix, with batching over explained rows. Estimated expectations retain telescoping reconstruction. Empirical monthly correlations also reflect conditional-integration error, reference approximation, realized finite samples, and shifts between the reference and forecast distributions.

## GaussianCOLA

```python
from orthogonal_explanations import GaussianCOLA

explainer = GaussianCOLA(
    x_reference,          # pretest transformed inputs
    fitted_model.predict,
    draws=16,
    baseline_draws=1024,
    batch_size=512,
    seed=20261023,
)
attributions, predictions = explainer.explain(x_forecast, seed=20261024)
baseline = explainer.base
```

`predict` must return one finite scalar per row. Reference and forecast arrays use identical transformed feature coordinates. The explanation path casts forecast inputs to float32, following Experiment 3; outputs use float64. Returned attribution columns retain the original input-feature identities. Set `order=[...]` to a permutation chosen before test evaluation. Reuse the same fitted explainer within its learning vintage.

For a scalar linear predictor, supply its coefficients in the same transformed coordinates and output units:

```python
explainer = GaussianCOLA(
    x_reference,
    fitted_model.predict,
    linear_parameters=(fitted_model.coef_, fitted_model.intercept_),
)
```

This uses exact Gaussian expectations and an exact linear baseline. For a target-standardized predictor, first express both the prediction callable and its coefficients in the original return units. The caller supplies the exact linear representation of that callable.

The reusable API is checked against Experiment 3's archived natural-order implementation for both exact-linear and nonlinear calculation paths. The case study implements its own analytical path on its original float64 inputs. Keeping these study-specific calculations in the standalone scripts preserves their numerical conventions.

## Coalition-table interface

`ordered_cola(values, k, order=None)` implements Experiment 1's increments from a precomputed coalition table. The table has one row per observation and $2^k$ columns; column `mask` is the estimated conditional value for that feature-bitmask coalition. Column zero is the baseline and the final column is the full prediction. This interface supports comparisons using the same coalition estimates across attribution methods.

## From arrays to reusable financial data

Keep asset, information date, realized-return date, model, feature names, and learning vintage beside every explanation vector. Store prediction, baseline, raw attributions, and the realized response separately. Preserve model-fit and preprocessing cutoffs when fitting multiple vintages. The [basic example](../examples/basic_usage.py) illustrates these identities; the [data notes](data.md) document each paper script's panel schema.

The downstream analyses in the paper condition inference on the fitted panels. Main empirical studies use pointwise bootstrap intervals and separate Holm test families; TWFE uses clustered inference. The frozen-predictor case study uses its own six-lag HAC and clustered procedures. Numerical conditioning and financial association are evaluated as separate properties of the resulting panel.
