# Data and explanation panels

## Source data

The scripts retrieve source data and record provenance in their output folders. Keep these records and the cached source payloads with a completed reproduction.

| Source | Use | Version in the supplied code |
| --- | --- | --- |
| [Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) | Monthly value-weighted 100 Size–Book-to-Market portfolios and monthly RF | Live `100_Portfolios_10x10_CSV.zip` and `F-F_Research_Data_Factors_CSV.zip`; downloaded bytes and hashes recorded by the scripts |
| [Adv-ALSTM](https://github.com/fulifeng/Adv-ALSTM) | Original 50 KDD17 adjusted-price instruments, including ADRs | Commit `3d6c3dc921f44832dd48734d79dbf60cd832cf5b` |
| [investment_data](https://github.com/chenditc/investment_data) | Adjusted CSI300 prices and historical constituent membership | Qlib archive from release `2026-10-24`; 569,216,239 bytes |
| [Federal Reserve H.10](https://www.federalreserve.gov/releases/h10/hist/dat00_ch.htm) | CNY per USD for CSI300 conversion | Historical daily observations selected by the source code |
| [FRED DEXCHUS](https://fred.stlouisfed.org/series/DEXCHUS) | Exchange-rate fallback | Requested daily dates 2006–2016 |

The expected CSI300 archive SHA-256 is:

```text
0628819ff65686ea31cc075af0aa1b896e2707b7c4e72b0269b3df209234124d
```

The reproduction downloads approximately 569 MB for that archive before extracting the required calendar, membership, and close-price files. CSI300 eligibility uses historical membership at the information date. US-priced KDD17 instruments use their supplied adjusted prices. Experiment 3 converts CSI300 prices to USD before building monthly returns. Month-end prices must lie within seven days of calendar month end, and adjacent prices must represent consecutive months.

French source archives and exchange-rate endpoints are live resources. Preserve each reproduction's cached bytes and generated source manifest to identify the exact input revision. Source datasets retain their own licensing and access conditions.

## Monthly variables

All six features use information available through information month $t$. The response is next-month USD excess return, $y_{i,t+1}=r_{i,t+1}-r_{f,t+1}$. Returns enter the code as decimal fractions; displayed coefficient/spread tables use basis points where indicated.

| Order | Feature | Definition from monthly total returns |
| --- | --- | --- |
| 1 | `Ret1` | Return in month $t$ |
| 2 | `Mom3` | Compounded return from $t-2$ through $t$ |
| 3 | `Mom12_2` | Compounded return from $t-11$ through $t-1$ |
| 4 | `Vol12` | Sample standard deviation over $t-11$ through $t$, with denominator 11 |
| 5 | `DownVol12` | Square root of the 12-month average of squared negative returns, using zero for positive returns |
| 6 | `Skew12` | Adjusted sample skewness over those 12 months |

Observations require complete feature histories and the outcome support required by each analysis. Transformations are fitted inside the historical chronological windows. The case study retains feature-eligible observations for its cross-sectional scaling and subsequently selects observed return outcomes for return regressions; survival analysis also records censoring at missing outcomes.

## Generated explanation panels

The study-specific export formats preserve their original names:

| Study | Location inside output directory | Identities and signal fields |
| --- | --- | --- |
| Experiment 1 | Per-case `explanations.csv` files under the simulation outputs | Synthetic observation identities and each explanation method's coordinates |
| Experiment 2 | `explanations/{model}/s{seed}/{YYYY-MM}.parquet` | Asset `id`, information `date`, `return_date`, `y`, `model`, `seed`, `model_year`, `pred`, `base`, and `A1`–`A6` |
| Experiment 3 | `explanations/{dataset}_{model}.parquet` | Dataset/model/asset/date identities, `prediction`, `baseline`, `seed_count`, `A_Ret1` through `A_Skew12`, and reconstruction diagnostics |
| Case study | `explanation_panel.csv` | `id`, `information_date`, `return_date`, `y`, `fit_cutoff`, `predictor`, `prediction`, `baseline`, `A_{feature}`, and standardized `Z_{feature}` |

For Experiment 2, map `A1`–`A6` to the ordered feature names using `tables/features.csv`. In Experiment 3, exported attributions are averaged across the fitted seeds for each dataset/model; annual-split and validation-selection tables retain learning-vintage provenance. Experiment 2 analyzes monthly model/seed statistics before aggregation. These aggregation choices match the supplied manuscript protocol.

The case study's `A_` fields contain raw return-unit attributions. Its `Z_` fields are standardized within each information-date cross-section; reported return-regression slopes are per attribution standard deviation. Preserve the raw fields when checking reconstruction.

The source release provides the generation code. Running the scripts creates the explanation panels, source manifests, tables, figures, fitted models, and inference diagnostics. Keep the complete output directory when sharing a completed reproduction, including the explanation files excluded from Experiment 2's default compact ZIP.
