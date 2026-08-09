# Reference and license audit

The machine-readable source of truth is [`references.lock.yaml`](../references.lock.yaml). No third-party source, checkpoint or dataset is vendored in this repository.

| Reference | URL and pinned revision | License/source status | Audited interface and actual use |
|---|---|---|---|
| DINOv3 | [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3), `6876159a11b4df116f30f667f8c9888617df0751` | DINOv3 custom license | External checkout/weights; local `torch.hub.load`, `forward_features()`, `get_intermediate_layers()` |
| TOPICTrack | [holmescao/TOPICTrack](https://github.com/holmescao/TOPICTrack), `e7b260f41a92fbe94419ce448aa7b53fe80a6814` | MIT | External checkout; `fast-reid/configs/bee/AGW_S50.yml` and vendored FastReID `build_model` only |
| FastReID | [JDAI-CV/fast-reid](https://github.com/JDAI-CV/fast-reid), `c9bc3ceb2f7a6438b62fb515ea3df6d1e999e95d` | Apache-2.0 | Config/model/checkpointer interface audit only; runtime uses TOPICTrack's pinned vendored snapshot |
| TrackEval | [JonathonLuiten/TrackEval](https://github.com/JonathonLuiten/TrackEval), `12c8791b303e0a0b50f753af204249e622d0281a` | MIT | MOTChallenge export and H3 HOTA/Identity/CLEAR metric-semantics audit; no source vendored and no runtime import |
| torchvision v0.22.1 | [pytorch/vision](https://github.com/pytorch/vision), `59a3e1f9f78cfe44cb989877cc6f4ea77c8a75ca` | BSD-3-Clause | Installed API `resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)` and matching official transforms |
| BEE24 | [official dataset page](https://holmescao.github.io/datasets/BEE24), no source revision | `LICENSE_NOT_STATED_BY_SOURCE` | User-lawfully-obtained MOTChallenge train/test trees; never redistributed |

The repository itself grants no top-level open-source license. Upstream notices and terms continue to govern their respective software and model weights. Before server use, the operator must independently confirm that their intended DINOv3 weight and BEE24 data usage is permitted.
