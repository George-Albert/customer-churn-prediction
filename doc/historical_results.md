# Historical aggregate assessment — private exercise

These figures describe a previous assessment using private labeled data. They
are not synthetic-demo results, independently reproducible public measurements
or evidence of production business impact. The original data, prompt and
individual predictions are not distributed.

| Earlier-fit June assessment | Value |
|---|---:|
| Eligible customers | 3,582 |
| ROC-AUC | 0.7003 |
| Balanced accuracy | 0.6507 |
| Accuracy | 0.6929 |
| Churn recall | 0.8199 |
| Retained recall | 0.4814 |

The procedure selected on May was an unweighted forest (200 trees, depth 6,
minimum leaf 20, seed 42) at threshold 0.59. June also includes customers scored
with a conditional fallback prior. These metrics belong to the January–April fit,
not the later full-label refit.

The AUC lead over the balanced forest on May was only about 0.00027; no
statistical superiority is asserted. Timing/target construction, availability,
annual seasonality, calibrated risk and intervention benefit remain unverified.
The synthetic demonstration may select another candidate or threshold.
