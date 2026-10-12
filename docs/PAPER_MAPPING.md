# Manuscript-to-code correspondence

| Manuscript component | Code | Implementation |
|---|---|---|
| Eq. 1 | `gsdvq.QUANTIFIERS` | Fixed order: few, a few, some, many, a lot of |
| Eqs. 2--4 | `distributions.ObservationEnsemble` | Frozen CountSE evidence; three separately initialized categorical heads; average probabilities |
| Count support, Appendix A.1 | `support`, `Config.nmax` | 0 through 100 and a distinct overflow event; counts above 100 go to overflow |
| Eqs. 5--6 | `ExpectationHead`, `prepare_features` | Frozen target-removed scene embedding concatenated with category embedding; original instance count supervises head |
| Eq. 7 | `DistributionComparator`, `numerics.gsd_statistics` | Maximum CDF violation over shared ordered support, including terminal overflow |
| Eqs. 8--9 | `DistributionComparator` | d+ = max(Fexp-Fobs), d- = max(Fobs-Fexp), sdom = d+ - d-, ainc = min(d+,d-) |
| Eqs. 10--12 | `QuantifierDecoder.logits` | Nonpositive signed slopes for few/a few; nonnegative slopes for many/a lot of; -gamma abs(sdom) for some |
| Decoder bottleneck | `GSDVQ.forward`, `QuantifierDecoder` | Exactly two comparator inputs; no pooled image features/category IDs reach the primary decoder |
| Eq. 13 | `participant_beta_nll` | One beta term per participant response, quantifier-specific learned positive concentration, boundary handling |
| Section 4.5 Stage A | `training.fit_auxiliary(stage="A")` | Count NLL plus CRPS, independently trained observation heads on decontaminated counting sources |
| Section 4.5 Stage B | `training.fit_auxiliary(stage="B")` | Count NLL plus CRPS on target-removed auxiliary images; no VAQUUM/human expectation supervision |
| Section 4.5 Stage C | `training.fit_decoder` | Frozen observation and expectation branches; beta NLL fits only decoder; validation categories select checkpoint |
| Section 5.2 | `data.category_folds`, `validate_fold` | Five category-disjoint outer folds; separate validation categories; participant responses stay attached to their item |
| Section 5.3 | `DistributionComparator`, `FrozenFeatureMLP` | Count, Context-CDF, strict PoS, directional mean + Wasserstein/JS, frozen-feature MLP control |
| Section 5.4, Appendix C | `predict_expectations`, `expected_count_report` | External human expected-count judgments used only in evaluation; item-wise normalization by human mean |
| Section 5.5 | `metrics.rating_report`, `transfer_report`, `count_report` | Five per-quantifier r/rho, macro means, rating MAE, preference JS, count MAE/RMSE/NLL/CRPS |
| Statistical protocol | `image_bootstrap`, `paired_permutation`, CLI seed summary | Image-cluster bootstrap 95% CI, paired swaps at image-object level, three-seed sample SD |
| Section 6.3 GT diagnostic | `--gt-observation-diagnostic`, `GSDVQ.forward` | Replace observation PMF by annotated-count delta only in explicitly marked diagnostic |
| Section 6.6 ablations | `configs/variants`, distribution/model flags | Point observation/expectation, no incomparability, single head, scene/category removal and mask variants |
| Section 6.7 | `diagnostics.disagreement_report` | Incomparability tertiles, human rating variance/entropy/error and partial rank correlations |
| Eq. 14, Appendix B | `decontamination.decontaminate` | Mutually exclusive source/exact then near filtering; dynamic audit counts satisfy accounting identity |
| Appendix D | Mask preparation and runtime `--mask-operator` | Mixed-mask auxiliary training and held-out-operator prediction using frozen heads/decoder |
| Appendix E | `FixedKPlan`, `fixed_k_mask`, `geometry_probe` | Training-only frozen K and decoy statistics, foreground-safe decoys, joint inpainting, mask-only count probe |

Zero-shot VLM controls require the original shared VAQUUM prompting protocol and
the relevant frozen model adapters. The supplied manuscript does not reproduce
that exact protocol; this package does not silently call a generic prompt by an
original baseline's name. Precomputed zero-shot scores can use the same evaluator.
