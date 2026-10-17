# Understanding and defending the workflow

1. **Define the unit and cutoff.** Customer-level groups avoid invoice-row leakage.
   Restrict observable history before calculating aggregates. Keep timing unknowns explicit.
2. **Inspect data and decide rules.** Diagnose dates, TENURE, identities, category
   tokens and amount signs before changing representations. Distinguish fallback from imputation.
3. **Fit preprocessing within learning.** Use medians and one-hot encoding; scale
   only logistic regression. Unknown categories are supported without learning from scoring data.
4. **Compare modest models.** Logistic regression, depth-limited tree and limited
   forest each have unweighted/balanced variants. No resampling or broad search.
5. **Choose the objective explicitly.** Full-May ROC-AUC selects ranking quality;
   May balanced accuracy selects an operating threshold. F1 depends on the cutoff
   and does not include true negatives. An always-churn reference can have high F1
   when churn is common. No metric is universally best or a substitute for costs.
6. **Assess a frozen procedure.** June includes usable and fallback customers.
   Scores from development are optimistic because they influenced selection.
7. **Refit only after assessment.** Expand learned vocabulary/statistics using all
   eligible labels; retain model settings and threshold. Final refit has no new holdout score.
8. **Validate the actual export.** Exact IDs, unique coverage, binary values and
   cohort-probability averaging are part of the prediction procedure.

References: [pipelines and leakage](https://scikit-learn.org/stable/common_pitfalls.html),
[threshold decisions](https://scikit-learn.org/stable/modules/classification_threshold.html),
[F1](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html),
[balanced accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html).
