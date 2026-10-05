# Method and evaluation notes

- The label indicates recorded disease during a single observation window.
  Exact onset times are unavailable. A current-status/window likelihood is a
  modeling choice, conditional on assumptions about detection and recording.
- Auxiliary labels mix occurrence indicators and terminal health states.
  Auxiliary labels are training supervision; baseline features and observation
  duration are the inputs at prediction time.
- The neural branch uses factorization-machine-style feature interactions.
  ICBoost uses the exact gradient of the window negative log-likelihood and
  positive Fisher-scoring curvature, not the observed Hessian.
- The main comparison uses 52 baseline features for the specified implementations.
  Models parameterize time differently. Ablations isolate individual components
  more closely than the full pipeline comparison does.
- Follow-up-stratified AUC pools within-quintile AUCs by positive–negative pair
  count. Strata with fewer than 10 observations or only one class are skipped.
  This reduces between-window ranking effects but retains within-stratum variation.
- The source function name `breslow_H0` is retained for compatibility. Its code
  fits an event-count-matched time scale and a monotone decile correction; it is
  not a conventional event-time Breslow estimator.
- Outer test subjects are isolated. Internal auxiliary-feature cross-fitting and
  tree cross-fitting use different partitions, creating dependencies within tree
  validation folds. Fusion and calibration reuse internal predictions. Fully
  nested representation learning and separate calibration are future refinements.
- The neural cumulative-risk head is monotone in time. The complete fused model
  does not impose global time monotonicity. Risk-profile figures concern the head.
- Residual bands use empirical quantiles of absolute binary-label residuals.
  Reported coverage is an empirical diagnostic, not a strict conformal guarantee
  or a confidence interval for a person's underlying risk.
- Corrected repeated-CV and LOYO comparisons are approximate because training
  sets overlap; chronological testing consists of one split. Pooled predictions
  across three CV repeats contain repeated observations of the same workers.
- Trajectory features use follow-up observations and are a retrospective
  sensitivity analysis, not a baseline deployable feature set.

Release preparation changes paths, entry points, documentation, and figure text.
It preserves the model fitting formulas and saved numerical results. Model
training was not repeated during repository packaging. The accompanying
validation record distinguishes executed checks from historical experiments.
