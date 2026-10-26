# Reproducibility

## Source provenance

The archived Colab export contains eight independent studies in 13,704 lines. Its SHA-256 is:

```text
f0dda01b4ce8c3d2e9311705895c8982cdd619d81cc596260c383868738f55a5
```

`tools/extract_experiments.py` selects each study's source range, replaces its notebook header and dependency-installation cell, and retains its scientific body verbatim. `archive/manifest.json` records the source ranges and generated-file hashes. Settings, feature order, model definitions, inference, figures, and output-writing code remain in each generated study. Regenerate with `python tools/extract_experiments.py`; verify with `--check`.

The reusable API is maintained separately in `src/orthogonal_explanations/`. The experiment scripts use their archived implementations so that packaging changes leave the supplied scientific calculations traceable. The archive also preserves historical preparation comments and notebook-specific setup instructions.

## Environment

Python 3.10 or newer is required. A clean environment can be prepared from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[experiments]'
```

For the scikit-learn and XGBoost versions explicitly pinned by the supplied appendix cells:

```bash
python -m pip install -r requirements-paper.txt
```

Those cells pin scikit-learn 1.6.1 and XGBoost 3.4.1. The remaining dependency ranges come from the supplied installation code and imports. Original completed-run environment files were absent from the supplied materials; runtime metadata produced by a reproduction should accompany its results. Record the complete environment with:

```bash
python -m pip freeze > environment-freeze.txt
```

The numerical/source checks for this release ran with Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.8.0, and pandas 2.2.3. Paper-version installation and full empirical reproduction have a separate validation scope. Changes in providers, numerical libraries, hardware, and fitting algorithms can affect numerical results and timings.

## Protocol retained from the supplied code

| Study | Sample and models | Repeats and inference |
| --- | --- | --- |
| Experiment 1 | 48 assets; six Gaussian AR(1) features; 60/12/60 train/validation/test months; correlations 0, 0.3, 0.6, 0.9; OLS/XGBoost/MLP | Seeds 20261023–20261025; 1,000 paired run resamples; 2,000 known-DGP calibration draws; 1,000 evaluation draws |
| Experiment 2 | French 100; expanding training from 1967; 60 validation months and final-month purge; realized-return test months 1992–2016 | Same three seeds for each model; 2,000 stationary-bootstrap draws; automatic mean block length 7 months |
| Experiment 3 | French 100, KDD17, CSI300; OLS/XGBoost/MLP; annual expanding estimation | OLS deterministic; three seeds for XGBoost/MLP; 10,000 bootstrap draws; automatic mean block length 4 months for the paper's 48-month window |
| Case study | French 100; fixed OLS predictor refitted using outcomes through November 2006; test return months 2007–2016 | Seed 20261025; analytical Gaussian COLA; six-lag HAC for monthly FM/QR; clustered TWFE and Cox inference |
| Appendix C | Synthetic and French 100 sensitivity studies | Same main model/seed protocols; 1,000 synthetic and 2,000 empirical bootstrap draws |

Experiment 3 requests evaluation from January 2012 and selects the first feasible common January using its observed support criteria. The manuscript's resulting window is January 2013–December 2016. Its `evaluation_calendar_candidates.csv`, `split_feasibility.csv`, and annual-split exports record that selection. Retain the automatic calendar rule when reproducing the supplied protocol.

The feature order is `Ret1, Mom3, Mom12_2, Vol12, DownVol12, Skew12`. Main empirical conditional integration uses 16 antithetic draws and 1,024 baseline draws. Experiment 1 uses 2,048 baseline draws. Experiment 3 and the case study evaluate linear-model expectations analytically. In the first two studies, linear expectations arise from the supplied antithetic integration implementation. Source and feature details appear in [data.md](data.md).

## Running and resuming

Run the commands in the root README from a common working directory so the appendix studies can discover main outputs. Settings appear near the top of each file in dictionaries such as `CFG`, `BS`, `DATA`, `RUN`, `INPUT`, `SPEC`, and `PLOT`. Change these in a separate experiment copy for exploratory work; the tracked scripts can be checked against the archive at any time.

In a local Python session, outputs go to the current working directory. In Colab, the original scripts prefer `/content` when that directory exists. Common output prefixes are:

| Script | Output directory prefix | Principal files |
| --- | --- | --- |
| 01 | `exp1_` + timestamp | `tables/main.csv`, `tables/main_ci.csv`, model-specific appendix tables, `figures/exp1.pdf` |
| 02 | `exp2_ff100_` + configuration fingerprint | `tables/main.csv`, `tables/appendix_main_by_model.csv`, `figures/exp2.pdf`, `explanations/` |
| 03 | `exp3_` + configuration fingerprint | `tables/analysis_summary.csv`, `tables/appendix_conditioning_by_specification.csv`, `tables/appendix_runtime_by_specification.csv`, `figures/exp3.pdf`, `explanations/` |
| 04 | `cola_case_study_` + timestamp | `explanation_panel.csv`, five table exports, numerical estimates, survival samples, native summaries |
| 05 | `appendix_reference_` + configuration fingerprint | `tables/simulation_summary.csv`, `tables/french_reference_summary_bp.csv` |
| 06 | `appendix_feature_order_` + configuration fingerprint | `tables/simulation_summary.csv`, `tables/french_matched_order_summary_bp.csv`, `tables/feature_orders.csv` |
| 07 | `appendix_integration_` + configuration fingerprint | `tables/integration_convergence_summary.csv`, `tables/french_integration_summary_bp.csv` |
| 08 | `appendix_downstream_` + configuration fingerprint | `tables/downstream_robustness_summary_bp.csv`, `tables/bootstrap_interval_width_sensitivity.csv` |

The output prefix for script 08 follows its source configuration; inspect the printed `Setup` path for the actual location. Appendix scripts also save their PDF/PNG figures and detailed numerical diagnostics. Figure and table numbers above refer to the supplied manuscript; extra diagnostics remain in the output folders.

Experiments 2–3 and the appendix studies use local checkpoints and configuration/source signatures. Rerunning the same file from the same working directory can reuse compatible checkpoints. Keep the complete output directories for model reuse. Set `RUN['output_dir']`, `RUN['cache_dir']`, or appendix `INPUT` paths explicitly where those settings are present. Script 08 can also reuse an existing downstream summary through its `PLOT` settings.

Headless execution supports `MPLBACKEND=Agg`. An example command is:

```bash
MPLBACKEND=Agg python experiments/02_financial_associations.py
```

Scripts create local ZIP archives at completion. Experiment 2's default compact ZIP omits its large explanation Parquet files; retain the full output directory or set `CFG['archive_explanations'] = True` before running. Experiment 3 includes explanation panels by default. Model caches and some checkpoints stay outside compact archives. Optional Colab browser-download calls execute only in that environment.

Experiment 1 contains an optional empirical-panel branch that checks `/content/panel.csv` by default. Its synthetic paper results run independently of that file. The supplied `REAL` dictionary describes optional user-panel columns and chronological splitting.

## Manuscript result mapping

Table 3 and Tables A1–A2 use experiment 01's aggregate and per-model attribution-quality/inference exports. Figure 1 and Table A3 use experiment 02's aggregate and per-model association exports. Figure 2 and Table A4 use experiment 03's compatibility, conditioning, and runtime exports. The case study creates five tables; the manuscript displays FM, quantile, and Cox results as Tables 4, 5, and 6, while its TWFE and portfolio-sort outputs support Section 6. Tables C1–C4 use scripts 05–08 in order. Tables 1–2 and B1–B2 describe algorithms and settings in the manuscript.

The scripts generate CSV and figure inputs. The manuscript's selected and rounded tables were assembled from those outputs. Reproduction should compare the appropriate aggregate, supported calendar, units, and rounding, rather than compare every diagnostic export with a single manuscript table.

## Validation performed for this release

The release checks verify the archived SHA-256, byte-identical regeneration of all eight scientific bodies, Python syntax, and the absence of notebook dependency-installation calls in the standalone scripts. Numerical tests compare the coalition-table API with Experiment 1 and compare analytical and nonlinear Gaussian COLA with Experiment 3 on fixed inputs and seeds. Additional checks cover reconstruction, exact linear prefixes, reference-distribution orthogonality, feature order, batching, and invalid inputs. The quick-start example is also executed.

Full-size model fitting, source downloads, paper tables, and performance timings require running the experiment scripts. The source release includes code and generation instructions; completed-run outputs were absent from the supplied attachments.
