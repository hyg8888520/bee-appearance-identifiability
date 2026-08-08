# BEE24-DINOTrack

Portable DINOv3 feature extraction and H1 cosine-retrieval queries for COCO-style BEE24 annotations. The repository contains source code, tests, documentation, dependency definitions, and an example config only. Dataset files, checkpoints, local configs, feature caches, and experiment results remain external.

## Server setup

```bash
git clone <repo>
cd BEE24-DINOTrack
pip install -e .
cp configs/h1_bee24.example.yaml configs/h1_bee24.local.yaml
# Edit dataset/checkpoint/output paths and experiment parameters.
python scripts/run_h1.py --config configs/h1_bee24.local.yaml
```

No Python file needs to be edited. `configs/*.local.yaml` is ignored by Git.

For a cheap preflight check that does not construct the model or run inference:

```bash
python scripts/run_h1.py \
  --config configs/h1_bee24.local.yaml \
  --validate-only
```

## Filesystem contract

The five concerns are deliberately separate:

| Concern | Location | Git policy |
|---|---|---|
| Source, tests, docs, dependency definitions | this repository | committed |
| Python/CUDA runtime | server environment created by `pip install -e .` | external |
| BEE24 dataset and annotation | `dataset.root`, `dataset.annotation` | external |
| DINOv3 checkpoint | `models.dinov3.checkpoint` | external |
| Feature cache and all experiment results | beneath `output.root` | external |

Every runtime path is read from YAML. Output artifact paths may be relative to `output.root` or absolute paths already beneath it; a path that escapes the root is rejected. Relative dataset, annotation, checkpoint, and output-root paths are resolved from the local config's directory.

Images always resolve as:

```text
dataset.root / images[i].file_name
```

For example, `/mnt/datasets/BEE24` and `BEE24-01/img1/000001.jpg` resolve to `/mnt/datasets/BEE24/BEE24-01/img1/000001.jpg`. Absolute or parent-traversing annotation `file_name` values are rejected.

## Startup validation

Before importing `timm` or constructing DINOv3, the runner checks:

- `dataset.root` is a directory;
- `dataset.annotation` is readable COCO-style JSON;
- at least one annotation image exists beneath the dataset root;
- `models.dinov3.checkpoint` is a file;
- `output.root` can be created and written;
- a requested CUDA index exists.

It prints a concise resolved-path summary and exits with code 2 plus actionable errors if validation fails. The selected image set is also checked in full before model construction.

## H1 outputs

The H1 baseline loads the configured local DINOv3 checkpoint through `timm`, extracts normalized global image features, caches them, and writes configured top-k cosine-similarity queries. All locations are explicit in YAML:

- `output.feature_cache`
- `output.query_results`
- `output.figures`
- `output.failure_cases`
- `output.metadata`

These paths are constrained to `output.root`. `figures` and `failure_cases` are prepared for downstream analysis; the global-retrieval baseline does not invent visualizations or failure criteria.

The example uses the `vit_small_patch16_dinov3.lvd1689m` model registered by `timm >= 1.0.20`. The local `.pth` remains outside the repository and is supplied to `timm` as the pretrained-weight file. Use a matching `model_name` when changing checkpoint architecture.

## Reproducibility and tests

The run metadata records the fully resolved config, feature shape, timings, cache status, and checkpoint SHA-256 for newly extracted features. Set `h1.reuse_feature_cache: false` when intentionally rebuilding cached features.

```bash
pytest -q
```

The test suite covers path composition, YAML resolution, output-root containment, startup failures, placeholder policy, checkpoint/local-config ignores, and absence of fixed Linux server paths in Python source.

