# Local data interface

All examples in `examples/*.jsonl` are **format examples**, not measured samples.
Convert the actual released dataset into the relevant manifest shape. Training
and prediction do not fetch images, annotations, model weights or human judgments.

## Records

Every item has `id`, `image_id`, `category`, `role` and `feature_file`. IDs must be
unique per image-object pair. Auxiliary examples also have `partition` (train,
validation or test), measured `count`, `dataset`, `image_file` and optional
namespace-qualified `source_ids` for audit. Expectation images need `mask_file`.

| Role | Use | Annotation fields |
|---|---|---|
| counting | Stage A, public counting resources | Original observed count |
| expectation | Stage B, COCO/LVIS target-removal construction | Number of original target instances before removal |
| vaquum | Stage C / inference | Training: `ratings` map each quantifier to participant responses; ordinary inference drops all ratings/count fields |
| human_expectation_validation | Independent evaluation only | Human expected_counts reside in a separate labels file; never used by training datasets |

VAQUUM training manifests join the input records with their `ratings` from the
separate annotation file. Keep every participant response; do not replace it by
the item mean during training. Normalize source ratings to [0,1] in the converter.
Prediction can use the same manifest, but its dataset reader does not load ratings.
Human validation scenes already edited for annotation can declare
`image_is_target_removed: true`; this prevents a second target-removal operation.

## Frozen feature caches

Use compressed `.npz` files without pickled objects. Required keys:

| Key | Shape | Meaning |
|---|---|---|
| counting | [Dcount] | Frozen text-guided CountSE evidence for original image/category |
| category | [Dcategory] | Frozen target-category text embedding |
| scene_primary_gray | [variants,Dscene] | Frozen scene features after gray target removal |
| scene_primary_blur | [variants,Dscene] | Frozen scene features after blurred target removal |
| scene_primary_inpaint | [variants,Dscene] | Frozen scene features after inpainted target removal |
| scene_fixed_k_inpaint | [variants,Dscene] | Fixed-K true masks + decoys, jointly inpainted |
| scene_validation | [Dscene] | Optional exact target-removed image displayed to human annotators |
| joint | [Djoint] | Frozen original image/text embedding, only for the MLP fairness control |

Only the keys needed for that stage are read. Stored extra arrays cannot bypass
the GSD bottleneck. Arrays must be finite and match configured dimensions. Save
each data/encoder/removal snapshot in its own immutable output directory.

## Local encoder preparation

Provide a factory implementing `FrozenEvidence`: counting(image, category),
scene(target_removed_image), category(category), and joint(image, category).
All encoders must use evaluation mode with gradients disabled. Record exact
checkpoint hashes, revisions and preprocessing in `encoder.provenance`.

```bash
python -m gsdvq.cli prepare --data data/raw_expectation.jsonl --image-root data --encoder-factory examples.frozen_encoders:LocalFrozenEncoders --encoder-kwargs '{"countse_checkpoint":"weights/countse.pt","scene_encoder_checkpoint":"weights/scene_text.pt"}' --inpaint-factory examples.frozen_encoders:LocalInpainter --inpaint-kwargs '{"checkpoint":"weights/inpaint.pt"}' --output-root features/expectation --manifest data/expectation_features.jsonl
```

The factory classes in that command are integration templates. Connect actual
installed local model wrappers before calling them; their methods deliberately
raise instead of inventing embeddings or painted images.

Instance masks are boolean `.npy` arrays [instances,H,W]; counts supervise the
expectation head before target removal. Optional foreground exclusion masks are
[H,W] boolean arrays including salient objects. Decoys require those masks.

## Decontamination and backbone provenance

```bash
python -m gsdvq.cli audit --auxiliary data/auxiliary_raw.jsonl --vaquum data/vaquum_images.jsonl --image-root data --filtered data/auxiliary_filtered.jsonl --output audits/auxiliary.json
```

Filtering is image-level; all category records for an image inherit its decision.
The output includes retained IDs, canonical image hashes, near-neighbor candidates
and mutually exclusive removal counts. An optional review file maps
`dataset:image_id` to `{duplicate: boolean, rationale: ...}`.

The auxiliary-data audit alone does not establish a pretrained backbone's
exposure. Strict training additionally requires a completed external backbone
pretraining audit, supplied by `--backbone-provenance`. It includes
`exposure_audit_status: audited`, `checkpoint_sha256` and supporting provenance.
Otherwise use the explicit diagnostic checkpoint policy. This implements the
manuscript's exposure distinction without asserting unverified decontamination.

Generate/cache features FIRST, then audit the final manifest used for training;
the strict audit verifies that exact manifest's SHA256. Auditing a raw manifest
and changing its feature paths afterwards creates a different snapshot.
