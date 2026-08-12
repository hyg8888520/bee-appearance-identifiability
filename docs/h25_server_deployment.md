# H2.5 server deployment

H2.5 只读取已完成 H2 的 signals 与 primary feature caches，通常不需要 GPU 重新抽特征。仍建议在 `beeid-dino` Python 3.11 环境运行，以保持依赖与 H2 一致。

## 1. 更新代码与创建配置

```bash
cd ~/projects/bee-appearance-identifiability
git fetch origin
git checkout codex/h25-refined-contamination-h3-protocol
git pull --ff-only
~/venvs/beeid-dino/bin/python -m pip install --no-deps -e .
cp configs/h25_smoke.example.yaml configs/h25_smoke.local.yaml
cp configs/h25.example.yaml configs/h25.local.yaml
```

`configs/*.local.yaml` 已被 Git 忽略。各路径含义：

- `bee24_root`：原始 BEE24 根目录；用于验证 H1 manifest contract，不写数据；
- `h1_output_root`：完成 H1 的目录，含 manifest、run metadata、cache registry；
- `h2_output_root`：完成 H2 的目录，含 signals、run metadata、H2 cache registry；
- `output_root`：新的 H2.5 输出目录，必须不同于 H1/H2；
- `cache_root`：H2 当时使用的 feature cache 根目录；registry 中的实际 cache 必须仍存在；
- 其余 repo/checkpoint 路径沿用 H2 配置。H2.5 不重新加载模型，但保留路径以验证完整 config contract。

`dataset.video_ids / max_frames_per_video / allow_subset` 必须与作为 source 的 H2 smoke/full 范围一致。full H2.5 使用空 `video_ids`、null limits 和两个 `allow_subset: false`。

## 2. 本地逻辑与真实数据 smoke

```bash
cd ~/projects/bee-appearance-identifiability
~/venvs/beeid-dino/bin/python -m beeid.cli h3-validate-protocol \
  --protocol configs/h3_protocol.lock.yaml \
  --checksum configs/h3_protocol.lock.sha256
~/venvs/beeid-dino/bin/python -m beeid.cli h25-synthetic-smoke \
  --output ~/experiments/bee-appearance-identifiability/h25-synthetic
bash scripts/h25_smoke_test.sh \
  configs/h25_smoke.local.yaml \
  ~/venvs/beeid-dino/bin/python
```

脚本故意不调用裸 `python`，以免落回 Conda base 或 `/usr/bin/python`。如果 H2 smoke source 不包含足够的连续历史或同帧 negative，事件数可能为 0；此时应增加 `max_frames_per_video`，不能伪造事件。

## 3. full development run

确认 smoke 后运行：

```bash
bash scripts/run_h25.sh \
  configs/h25.local.yaml \
  ~/venvs/beeid-dino/bin/python
```

若 official AGW cache 在 H2 registry 中有效，可一起比较，但始终标记 reference-only。如果不希望在主分析展示 AGW，可直接组合运行：

```bash
~/venvs/beeid-dino/bin/python -m beeid.cli h25-run-all \
  --config configs/h25.local.yaml \
  --models resnet50 dinov3 \
  --confirm-full
```

中断后重复同一命令即可；输出采用原子写入，输入 cache 为只读。不要运行或读取 final test。完成后优先审查 `h25_paired_summary.csv`、`h25_cluster_bootstrap.csv`、`h25_run_metadata.json` 和 `h25_figures/`。
