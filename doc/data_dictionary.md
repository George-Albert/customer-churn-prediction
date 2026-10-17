# Data dictionary and representation

The generator emits a source-shaped interface to exercise the analysis.
Identifiers are fictional strings prefixed SYN-. Columns normalize to lowercase
in both loaded datasets; input bytes stay unchanged. Literal NA is a category.

| Source field | Interpretation in this demonstration |
|---|---|
| PK_INVOICES | Invoice identifier; several product rows may share it |
| FK_DATE_INVOICE | Checkout date as YYYYMMDD text |
| CUSTOMER_START_DATE | First day of acquisition month |
| TENURE | Customer-month index: 0 first month, 1 second |
| FLG_DWH_CUSTOMER_IS_BUSINESS | Binary business/private flag |
| FK_DWH_CUSTOMERS | Customer identity, never a model predictor |
| FK_SUBSCRIPTIONS | Stable customer/product subscription identity |
| PRODUCT_GROUP / PRODUCT_SUBCATEGORY | Product categories; source naming is preserved |
| mrr | Signed fictional record amount; no asserted currency/accounting meaning |
| ORIGIN_GROUP / ORIGIN_SUBGROUP | Acquisition categories |
| noise | Synthetic binary target in labeled inputs only |

Features summarize month coverage, distinct activity, signed record amounts,
product presence and month-to-month changes. No month is silently filled with
zero. Median imputation occurs inside the fitted pipeline. Product vocabulary
comes from usable learning customers, then all usable labels for final refit.
Dates, IDs, target, partitions and fallback flags never become predictors.
