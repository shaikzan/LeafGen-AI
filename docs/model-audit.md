# LeafGen AI model and experiment audit

Audit date: 2026-11-28

This audit inspects the checkpoints and inference code present in the repository and the published model repository. It does not establish field accuracy or independently reproduce the training experiment.

## Repository evidence

- The repository contains no PlantVillage dataset, training notebook/script, split manifest, or per-image test predictions. The existing `test_prediction.py` expects `backend/test_images/tomato_leaf.jpg`, which is not present.
- The public model repository contains `cnn_real.pt`, `cnn_aug.pt`, `generator.pt`, `classes.json`, `rare.json`, `results.json`, and PNG result figures. The downloader now includes both classifier variants and `results.json`.
- The checkpoints are never silently replaced by randomly initialized models. ResNet18 uses strict state-dict loading and validates its 38-output classifier head.

## Classifier checkpoint and mapping

- `cnn_real.pt` loads strictly into torchvision ResNet18. Its `fc.weight` shape is `(38, 512)`; the 122 state entries load with no missing or unexpected keys.
- `cnn_aug.pt` is now present locally and also loads strictly into the same architecture. It has a 38-output head and different learned classifier weights.
- The local `classes.json` contains 38 unique entries. The rare indices `[2, 36, 7, 1, 0, 31]` resolve to the six rare classes listed in the project notes.
- The backend exposes `GET /api/classes`; the Synthetic Studio now consumes that exact ordered mapping instead of maintaining its own class list.
- This proves checkpoint dimensions and internal mapping consistency. The original training code is absent, so the original `ImageFolder.class_to_idx` mapping cannot be independently compared to the checkpoint. The published class list is the available authority.

## Inference behavior

Inference in `leafgen_model.py` resizes RGB images to 64×64 with `INTER_AREA`, applies CLAHE to LAB luminance (clip limit 2.0, 4×4 tiles), converts back to RGB, scales to `[0,1]`, bilinearly upsamples to 128×128, applies ImageNet mean/std normalization, and runs ResNet18 in evaluation/inference mode. Softmax is applied once over all 38 logits. The complete vector is checked for finite values and a sum of 1 before top-k selection.

The training code is unavailable, so preprocessing parity is supported by the model helper's explicit training note and supplied project description, but cannot be confirmed against the actual training transforms. ImageNet normalization and the 128×128 step therefore remain unverified against original training code.

The repeated external `Strawberry - Leaf scorch` predictions cannot be reproduced from this checkout: the referenced external photographs are not included. The behavior is not evidence by itself of a bad checkpoint or class order. Mapping uniqueness, matching head size, strict checkpoint loading, and preprocessing integrity checks pass; determining whether domain shift or another issue caused those specific predictions requires the exact images and labeled held-out data.

Raw softmax values are not calibrated probabilities or an out-of-distribution detector. The existing 0.70 warning threshold is only a heuristic. This audit does not claim non-leaf detection or calibrated uncertainty.

The prediction response now includes predicted crop and disease, `raw_model_score`, `score_type`, `calibrated_score: null`, `input_validity_status: not_checked`, and rejection/uncertainty status. The legacy `confidence` field remains for frontend compatibility. Low-score warnings say the threshold is heuristic; they do not reject input.

No trained leaf/non-leaf checkpoint or labeled binary dataset exists in the repository. A public leaf-phenology model was considered, but its published scope is phenological states for deciduous woody plants and explicitly excludes herbaceous plants, so it is not appropriate as a general leaf-validity gate for these crops. No detector is fabricated or claimed.

## DCGAN checks

- The conditional generator checkpoint loads strictly into the shipped generator: 100-dimensional noise plus a 38-way one-hot label, transposed-convolution generator, 64×64 RGB `tanh` output, mapped to uint8 `[0,255]` for PNG responses.
- Five requested class conditions produced finite, nonconstant, distinct images. A sample inspection showed a recognizable leaf-like central shape, but images are coarse/blurry at 64×64 with simple backgrounds.
- As a small, non-independent sanity check only, the real-only classifier top-1 matched the requested condition on 4 of 10 generated samples across Apple scab, Grape black rot, Tomato late blight, Strawberry leaf scorch, and Apple healthy. This is not a validated GAN quality score; the classifier itself is imperfect on synthetic data. It does show that successful conditioning calls do not guarantee semantic class fidelity.
- No FID, diversity benchmark, or independently labeled synthetic-image evaluation is available. Class-conditioned generation is wired and functional; class correctness and broad diversity remain unproven.

## Published experiment arithmetic

The downloaded `results.json` contains three run summaries for each condition. Recomputing the means from its per-run metrics reproduces the published summary:

| Metric | Real only mean | Real + synthetic mean |
|---|---:|---:|
| Macro-F1 | 0.9783746 | 0.9708397 |
| Accuracy | 0.9842049 | 0.9777805 |
| Mean recall, six rare classes | 0.9356337 | 0.9401673 |

The rare-class recall difference is +0.00453, while the JSON reports run standard deviations of 0.01683 (real-only) and 0.02085 (augmented). Thus the wording that the small increase is within observed seed variation is consistent with the supplied run summaries. The macro-F1 and accuracy decreases also match the website's rounded values.

This verifies the arithmetic in the published JSON only. The dataset, split IDs, training notebook, and raw predictions are absent, so leakage checks, metric recomputation from predictions, the reported 8,146-image test count, and end-to-end experiment reproducibility remain unverified.

## Automated checks

`python backend/test_model_integrity.py` covers both classifier checkpoints, the 38-class mapping and rare indices, full probability vector shape/finite normalized values, deterministic top-three inference, invalid image arrays, conditional generator output dimensions/range/diversity, service image encoding, corrupted and empty image rejection, the decoded-dimension guard, and evaluation report generation. All ten checks passed in the audited environment.

`backend/evaluate_classifier.py` evaluates a labeled folder tree using the real-only checkpoint and writes per-image predictions, top-three scores, expected/predicted labels, correctness, validity status, accuracy/top-three accuracy, macro metrics, per-class recall, confusion matrix, and prediction distribution. It supports separate `held_out` and `external` runs. The repository contains no labeled held-out or external image directory, so only evaluator plumbing was smoke-tested; no disease accuracy metrics were generated from the synthetic smoke image.

These are model/inference integrity checks, not an accuracy evaluation on PlantVillage or real-world images.
