# Orthogonal Explanations

**Orthogonal Explanations for Machine Learning in Financial Econometrics**  
Dongseok Kim · Research code accompanying the manuscript

[Quick start](#quick-start) · [Paper experiments](#paper-experiments) · [Reproducibility](docs/reproducibility.md) · [Data](docs/data.md) · [Method and API](docs/method.md) · [License](LICENSE)

This framework organizes out-of-sample local explanations into reusable asset–month panels. Conditional Orthogonal Local Attributions (COLA) decompose predictions in a prespecified feature order. The resulting signals support Fama–MacBeth regressions, two-way fixed-effects regressions, quantile regressions, portfolio sorts, and a drawdown-timing case study.

## Method at a glance

1. Fit preprocessing and a prediction model on chronological historical windows.
2. Construct local attributions while preserving asset, date, feature, and learning-vintage identities.
3. Reuse the same explanation panel across complementary financial analyses.

For a fitted predictor and reference distribution, each COLA coordinate is the increase in conditional expected prediction after observing the next feature. The increments reconstruct the prediction relative to its baseline. Exact conditional expectations give mutually orthogonal increments under the reference distribution. See the [method notes](docs/method.md) for the reference assumptions and finite-sample interpretation.

## Quick start

With Python 3.10 or newer, run from the repository root:

```bash
python -m pip install -e .
python examples/basic_usage.py
```

The [small example](examples/basic_usage.py) fits a synthetic linear predictor and constructs an out-of-sample explanation panel. The reusable API lives in [`src/orthogonal_explanations/`](src/orthogonal_explanations/).

## Paper experiments

Install the complete experiment dependencies once:

```bash
python -m pip install -e '.[experiments]'
```

| Script | Study | Manuscript results |
| --- | --- | --- |
| [01_explanation_quality.py](experiments/01_explanation_quality.py) | Correlated synthetic inputs; attribution quality and common-effect inference | Table 3; Tables A1–A2 |
| [02_financial_associations.py](experiments/02_financial_associations.py) | French 100, 1992–2016; return, tail, sort, and volatility-state associations | Figure 1; Table A3 |
| [03_cross_dataset_compatibility.py](experiments/03_cross_dataset_compatibility.py) | French 100, KDD17, CSI300; OLS, XGBoost, MLP; four analyses | Figure 2; Table A4 |
| [04_case_study.py](experiments/04_case_study.py) | Frozen OLS explanation panel, 2007–2016; five analyses | Tables 4–6; Section 6 |
| [05_conditional_reference.py](experiments/05_conditional_reference.py) | Full versus diagonal Gaussian references | Table C1 |
| [06_feature_order.py](experiments/06_feature_order.py) | Six prespecified feature orders | Table C2 |
| [07_integration_budget.py](experiments/07_integration_budget.py) | Conditional draws and antithetic sampling | Table C3 |
| [08_downstream_specifications.py](experiments/08_downstream_specifications.py) | Bootstrap blocks, quantiles, sorts, and state definitions | Table C4 |

Run any script independently, for example:

```bash
python experiments/01_explanation_quality.py
python experiments/02_financial_associations.py
python experiments/03_cross_dataset_compatibility.py
python experiments/04_case_study.py
python experiments/05_conditional_reference.py
python experiments/06_feature_order.py
python experiments/07_integration_budget.py
python experiments/08_downstream_specifications.py
```

The scripts preserve the supplied settings: three run seeds, OLS/XGBoost/MLP, and the original integration and bootstrap budgets. Empirical studies download their source datasets; the CSI300 archive is approximately 569 MB. Training and the appendix analyses can take substantial time. Running the main studies first lets the appendix scripts discover compatible local outputs and reuse models. The [reproducibility notes](docs/reproducibility.md) explain output locations, resuming, table mapping, and version settings.

## Source and reproducibility

[`archive/orthogonal_explanations.py`](archive/orthogonal_explanations.py) preserves the uploaded Colab export byte for byte. Eight scripts in [`experiments/`](experiments/) retain each study's scientific body and use explicit dependency installation through `pyproject.toml`. [`tools/extract_experiments.py`](tools/extract_experiments.py) verifies the archive hash and regenerates the scripts and their [manifest](archive/manifest.json).

```bash
python tools/extract_experiments.py --check
python -m unittest discover -s tests -v
```

Release validation covers all eight scripts' extraction and syntax, the example, and ten numerical/source checks, including agreement of the reusable API with the archived calculations. Full paper experiments and source downloads remain separate reproducibility runs. Explanation panels, fitted models, tables, and figures are generated locally by the scripts; this source release includes their generation procedures.

## Citation

```bibtex
@misc{kim2026orthogonal,
  author = {Kim, Dongseok},
  title = {Orthogonal Explanations for Machine Learning in Financial Econometrics},
  year = {2026},
  note = {Manuscript and accompanying research software},
  howpublished = {\url{https://github.com/AndrewKim1997/Orthogonal_Explanations}}
}
```

Software metadata is also provided in [`CITATION.cff`](CITATION.cff).

## License

Released under the [MIT License](LICENSE). Source datasets retain their providers' terms; the scripts retrieve them from the sources described in [Data](docs/data.md).
