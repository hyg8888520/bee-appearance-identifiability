# Target-server runbook

Target profile supplied for this project: RTX 4090 24 GB, 16 allocated Xeon cores, 120 GB RAM, NVIDIA driver 580.105.08, maximum driver CUDA 13.0. Linux distribution and scheduler are not yet confirmed.

## Preconditions

- Repository is public and contains no credentials, data, checkpoints, caches, or experiment outputs.
- BEE24, checkpoints, third-party checkouts, cache and output are outside the repository.
- DINOv3 and TOPICTrack are checked out at revisions in `references.lock.yaml`.
- `configs/h1.local.yaml` contains all paths, including both environment interpreters.
- No driver change, source-built PyTorch, Git LFS data or old TOPIC tracking environment is required.

## Installation and evidence order

Use `scripts/setup_dino_env.sh` and `scripts/setup_topic_env.sh`. Both request official stable cu128 wheels. Then run:

```bash
source /path/to/beeid-dino/bin/activate
bash scripts/check_server_env.sh configs/h1_smoke.local.yaml
python scripts/gpu_smoke.py --config configs/h1_smoke.local.yaml
beeid synthetic-smoke --output /path/outside/repo/synthetic-smoke
bash scripts/smoke_test.sh configs/h1_smoke.local.yaml
```

`server-check` prints and verifies `torch.__version__`, `torch.version.cuda`, CUDA availability, device name/capability, external paths and pinned checkouts. `gpu_smoke.py` executes a CUDA matrix operation and one real crop through ResNet50, DINOv3 `forward_features()`/AMP and strict AGW/AMP. Save console logs together with generated JSON.

Next run one full video, call `beeid estimate` with measured throughput/dimension/peak memory, and review predicted time/disk. Only after explicit user confirmation run full H1. `run_h1.sh` supplies the required full-run confirmation flag; avoid calling it during preflight. `resume_h1.sh` revalidates each feature shard before reuse.

The first full H1 development run has been completed and audited; this runbook remains the acceptance procedure for a fresh environment or rerun. H2 server execution is documented separately in `docs/h2_server_deployment.md`. No Slurm example is supplied while scheduler type is unknown.
