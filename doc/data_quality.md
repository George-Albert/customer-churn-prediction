# Quality decisions and limitations

## Implemented decisions

- Use TENURE 0/1 **and** checkout in [start, start + two calendar months).
- Preserve negative/tiny amounts and literal NA; source semantics are unknown.
- Exclude mixed-cohort and shared labeled identities. Preserve all scoring IDs.
- Retain a registry of customers without usable rows and score them with a
  conditional learning-fallback prior; do not invent zero-activity histories.
- Average cohort probabilities equally before thresholding an ambiguous scoring ID.
- Split acquisition cohorts: January–April learn, May select, June assess.
- Never fit preprocessing or numeric priors on May, June or scoring rows during
  selection. Final refit uses all eligible labels after the fixed assessment.

## Timing and business interpretation

The proposed scoring point is after the two acquisition calendar months. The
label has an acquisition-anchored twelve-month reference. This is classification,
not time-to-event estimation. The nominal gap is not a verified warning before
individual churn. Event/censoring dates, historical availability and active-at-cutoff
status are unavailable. A retrospective cohort split is not a deployable historical
backtest when twelve-month outcomes would not yet have matured.

Usage disengagement is not automatically contractual churn. Missing invoices
alone do not establish silent disengagement. Actionable lead time, annual
seasonality, calibration and retention-intervention benefit are not demonstrated.
The fictional demo provides no evidence for these business properties.

## Public versus historical evidence

All generated public inputs, displayed demo scores and demo figures are synthetic.
See [historical results](historical_results.md) for separately labeled aggregate
metrics from the private exercise. Neither individual source records nor original
predictions are included.
