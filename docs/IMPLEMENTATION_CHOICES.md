# Explicit conventions for underspecified details

The manuscript defines methods, training objectives, data separation, five folds
and three seeds. Appendix A.2 explicitly states that optimizer-specific numbers
and executable configurations are absent. Values below fill those gaps as
reference choices; they are not presented as original experiment settings.

- Count heads: two-layer GELU MLP, hidden width 128. Frozen feature dimensions
  default to 512 until replaced by actual local encoder dimensions.
- Optimizer: AdamW, learning rate 1e-3, weight decay 1e-4, batch size 64. Maximum
  Stage A/B/C lengths are 30/30/100 epochs with patience 10. Stage A/B select
  minimum validation NLL+0.1 CRPS; Stage C selects validation macro Pearson r.
- Seed values: 42/43/44; outer-fold shuffle seed 2026. Validation categories occupy
  approximately 20% of remaining categories. Explicit original folds can replace
  these generated folds without modifying the method.
- Head initialization is separate per ensemble member using independent random
  draws. Each head receives its own NLL+CRPS loss. Probabilities are averaged for
  inference, never logits. All large-encoder features stay frozen and cached.
- Boundary ratings are clamped to [1e-4,1-1e-4]. Concentration uses softplus plus
  1e-4, initialized at 10. The participant likelihood averages over actual valid
  responses, not padded entries or item means.
- GSD uses the ordered overflow event directly; the final CDF difference is set
  to zero to avoid roundoff. No numerical threshold or learned comparator replaces
  Eqs. 7--9.
- The overflow event's representative numeric value defaults to 101 for moments
  and distance diagnostics. It is a lower-bound proxy, not recovered tail count.
  Report true high-count errors with this convention or supply a defensible tail
  representative; exact raw-count moments cannot be inferred from a single event.
- Point-distribution ablations round the predicted mean to the common support.
  Context-CDF evaluates the expected CDF there. PoS is strict superiority without
  counting ties. Count/CDF/PoS one-statistic inputs are centered and padded with
  a zero second coordinate. Directional Wasserstein/JS receive normalized signed
  mean difference plus their distance and use the same constrained decoder.
- Gray, blur and inpainting are sampled uniformly in Stage B. There are three
  cached boundary variants per operator; boundary perturbation is outward dilation
  by 0--3 pixels to preserve full target coverage. Blur radius is 15. Inpainting
  requires a real external adapter and is never replaced by blur under that name.
- Primary Stage C and prediction use a deterministic inpainted realization (first
  cache variant). Operator transfer chooses gray/blur/inpaint at inference while
  leaving the trained weights fixed. This evaluation operator choice is configurable.
- Fixed-K uses the 99th training-count quantile, rounded upward. Decoy ellipse
  area/aspect statistics come from actual training instance masks. All true mask
  pixels remain covered; decoys cannot touch salient foreground or other edited
  components. The final binary edit union must contain exactly K components.
- Geometry-infeasible examples are explicit failures. If a cohort is restricted
  for this reason, restrict EVERY compared diagnostic variant to that same cohort
  and report exclusions. Do not shrink targets or silently clip component count.
- Fixed-K auxiliary features are jointly inpainted. Its config therefore uses the
  inpaint operator alone, with the same head dimensions, seeds and optimizer.
  Reuse the original observation checkpoint; retrain only expectation then decoder.
- Mask-only probe: ridge regression on resized binary mask pixels, never RGB,
  category identity or scene features. The geometry probe and test metrics are
  exposed as functions to accept the actual matched train/test cohorts.
- Human nCRPS is average proper CRPS over individual expected-count judgments,
  normalized item-wise by their mean. nW1 compares predicted and empirical CDFs,
  with the same normalization. Zero human-mean cases have undefined normalization
  and require separate reporting; they are never silently divided by an arbitrary
  epsilon. Overflow is subject to the same declared support convention.
- Production/MC transfer normalizes the SAME sigmoid acceptabilities over eligible
  quantifiers; it never fits a new transfer head. The exact original VAQUUM
  normalization is not given in this manuscript, so this is a configurable adapter
  convention to replace if the original protocol differs.
- Rating MAE averages available image-object-quantifier item means. Correlations
  are per quantifier across items and macro-averaged over all five; undefined
  correlations remain null. All five must be defined for macro scores.
- Disagreement uses participant population variance, ten-bin rating entropy,
  equal-sized incomparability tertiles and rank-residual partial correlation.
  These entropy/grouping details are reference conventions, not recovered settings.
- pHash uses 32x32 grayscale DCT, 8x8 low frequencies and Hamming radius 6.
  Unreviewed neighbors are conservatively removed; reviewed decisions are retained
  in the audit. Source/exact and near removals are sequential and disjoint.
- Bootstrap repeats default to 2000; permutation repeats to 5000. All category
  folds are merged per seed before primary evaluation; sample SD is taken over
  three complete seed reports, not over the five folds.

All reported experiment numbers must come from actual persisted predictions.
The supplied package contains no prefilled result tables or claimed reproduction.
