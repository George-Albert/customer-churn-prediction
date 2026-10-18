# Customer Churn Prediction

A reproducible customer-level workflow for turning early invoice histories into
a binary churn decision: audit the inputs, define the observation window, build
features, compare simple models, evaluate a frozen procedure and validate the export.

**The public demonstration uses fictional data generated from scratch.** It
contains no original customer records, private exercise prompt, individual source
predictions or fitted private models. This is an independent portfolio project,
not an official company implementation or a deployed retention system.

## Start here

- [Executed notebook with visible tables and figures](doc/demo_results/customer_churn_demo.ipynb)
- [Runnable synthetic notebook](notebooks/customer_churn_demo.ipynb)
- [Problem and output contract](doc/assignment_brief.md)
- [Quality decisions and timing limits](doc/data_quality.md)
- [Feature dictionary](doc/data_dictionary.md) and [methodology guide](doc/study_guide.md)
- [Separately labeled historical aggregate results](doc/historical_results.md)

## Reproduce the demonstration

Verified stack: Python 3.12.14 with the versions in `requirements.txt`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -q
python -m src.generate_demo_data
jupyter nbconvert --execute --to notebook notebooks/customer_churn_demo.ipynb --output customer_churn_demo.executed.ipynb --output-dir outputs --ExecutePreprocessor.timeout=600
python -m src.export_demo_results
```

For interactive use, select this environment's kernel and run all cells from the
repository root or `notebooks/`. The notebook generates demo inputs and executes
the full modular workflow. It displays aggregate results and five figures.
Individual synthetic predictions are written to `outputs/submission/predictions.csv`
with exactly `FK_DWH_CUSTOMERS` and integer `Predictions` (0/1). All generated
inputs, row-level outputs and models stay outside Git. The final command refreshes
only the curated aggregate snapshot in `doc/demo_results/`; the source notebook
remains output-free. Read the snapshot online; rerun the source in `notebooks/`.

Generation is deterministic at seed 42. The generator refuses to overwrite
unrecognized existing input data. To keep Jupyter caches local when needed:

```bash
export JUPYTER_CONFIG_DIR="$PWD/.cache/jupyter/config"
export JUPYTER_DATA_DIR="$PWD/.cache/jupyter/data"
export JUPYTER_RUNTIME_DIR="$PWD/.cache/jupyter/runtime"
export IPYTHONDIR="$PWD/.cache/ipython"
```

## Design and decision rules

| Stage | Decision |
|---|---|
| Unit | Customer/cohort, not invoice row; IDs never predict |
| Eligibility | TENURE 0/1 and checkout before start + two calendar months |
| Features | Month coverage, activity, signed amounts, products and changes |
| Learning | January–April fit vocabulary, preprocessing, models and fallback prior |
| Development | May selects among six fixed candidates by full-population ROC-AUC |
| Threshold | Maximize May balanced accuracy on a fixed 0.01 grid |
| Assessment | June scores the frozen procedure, including fallback customers |
| Final fit | Relearn statistics on all eligible labels, keeping selected settings |
| Export | Average cohort probabilities before one binary decision per customer |

Candidate families are logistic regression, a depth-limited decision tree and a
limited random forest, each with and without balanced class weights. A constant
majority reference anchors the comparison. Only logistic regression is standardized.
No SMOTE, deep learning, broad hyperparameter search or probability calibration is used.

ROC-AUC assesses discrimination without choosing a single cutoff. F1 combines
churn precision/recall at a cutoff and omits true negatives. Balanced accuracy
gives churn and retained recall equal weight. These objectives are declared, not
claims to optimize unknown retention costs or causal intervention benefit.

## Visible synthetic results

The seed-42 demonstration selected `forest_unweighted` on May ROC-AUC and chose
threshold **0.40** on May balanced accuracy. June then assessed the frozen
procedure on **200 fictional customers**, including 20 fallback customers.

| June synthetic assessment | Selected forest | Majority reference |
|---|---:|---:|
| ROC-AUC | 0.6823 | 0.5000 |
| Balanced accuracy | 0.5939 | 0.5000 |
| F1 (churn) | 0.7394 | 0.0000 |
| Churn recall | 0.9545 | 0.0000 |
| Retained recall | 0.2333 | 1.0000 |

High churn recall comes with many false positives: **69 retained customers** are
flagged and **5 churn customers** are missed. This illustrates why F1 alone does
not establish a useful retention policy. The reference predicts retention for
everyone, following the learning cohort's majority class.

![Synthetic June ROC curve](doc/demo_results/evaluation_roc.png)

![Synthetic June confusion matrix](doc/demo_results/evaluation_confusion.png)

After assessment, the final fit used 1,080 eligible fictional histories and
exported one binary prediction for each of **241 unique scoring IDs**, including
24 fallback customers. This coverage check is not an evaluation of final-fit
accuracy. [Aggregate values and provenance](doc/demo_results/summary.json),
[all five figures and reading guide](doc/demo_results/README.md), and the
[executed notebook](doc/demo_results/customer_churn_demo.ipynb) are included.
These metrics describe synthetic mechanics only; historical private results
remain in their separately labeled document.

## Evidence and limits

Synthetic patterns and metrics demonstrate software mechanics, not business
performance. The private exercise previously achieved June ROC-AUC 0.7003 and
balanced accuracy 0.6507; those earlier-fit aggregates are documented separately
and are not claims about this synthetic run or a production system.

A two-month observation window does not prove actionable lead time or historical
availability. The supplied fixed-horizon classification design is not survival
analysis. Missing event dates, label maturity, active-at-cutoff status and usage
logs prevent claims about advance warning, silent disengagement, annual
seasonality, calibrated risk or retention uplift.

## Repository map

| Directory | Purpose |
|---|---|
| `src/` | Generator, audit, features, model selection, assessment and export |
| `tests/` | Synthetic contracts, leakage guards and reproducibility tests |
| `notebooks/` | Explanatory public demo using the same modules |
| `doc/` | Decisions, interpretation and curated synthetic results in `demo_results/` |
| `input/` | Locally generated demo inputs; only its README is tracked |
| `outputs/` | Regenerable local results and models; ignored |

Every stage records input/output hashes and installed versions. No untrusted
model files are loaded. Model serialization is not canonical across processes:
compare numerical state/predictions and each run's manifest rather than requiring
cross-process joblib byte equality. Dependencies and historical records are not
redistributed under a new license; no software license is added in this release.
