# H5 服务器运行

H5 复用已经完成的 H3 feature cache（特征缓存）和 `h3_assignments.csv`，不会再次运行 DINOv3/ResNet50 特征提取。训练的是一个小型关联头，因此 GPU 显存预计远低于 backbone 训练；利用率可能随逐视频变长张量而波动。

```bash
cd ~/projects/bee-appearance-identifiability
git checkout codex/h5-beetrackquery
~/venvs/beeid-dino/bin/python -m pip install --no-deps -e .

cp configs/h5.local.yaml.example configs/h5.local.yaml
# 只编辑外部路径与 runtime 资源；h3_output_root 指向已完成的 H3 GT-box 目录。

~/venvs/beeid-dino/bin/python -m beeid.cli h5-validate-protocol \
  --protocol configs/h5_protocol.lock.yaml \
  --checksum configs/h5_protocol.lock.sha256
~/venvs/beeid-dino/bin/python -m beeid.cli h5-synthetic-smoke \
  --output ~/experiments/bee-appearance-identifiability/h5-synthetic
```

真实 subset smoke 需要 H3 source cache 覆盖同一组 train/dev 视频：

```bash
cp configs/h5_smoke.example.yaml configs/h5_smoke.local.yaml
bash scripts/h5_smoke_test.sh configs/h5_smoke.local.yaml ~/venvs/beeid-dino/bin/python
```

确认 smoke 后运行全量 development：

```bash
bash scripts/run_h5.sh configs/h5.local.yaml ~/venvs/beeid-dino/bin/python
```

H5 v2 不再对每个 clip 单独启动一次 optimizer step，而是批量处理相邻帧 transition。`runtime.batch_size` 是 transition 数上限；冻结参数 `max_pair_elements_per_batch` 进一步约束 padding 后的 pair 张量，避免拥挤帧造成显存峰值。首次运行只扫描一次压缩 H3 shards，并在仓库外的 H5 输出目录生成按 observation 对齐的 `.npy` cache；后续跟踪直接复用，不再重复扫描全部 shard。该连续 cache 预计额外占用约 2.7 GB（以当前 276,687 条 observation、ResNet50 2048 维和 DINOv3 384 维估算）。

在第二个终端监控 cache 对齐、epoch/batch、ETA、最近持久化断点、GPU 利用率和显存：

```bash
bash scripts/monitor_h5.sh \
  configs/h5.local.yaml \
  ~/venvs/beeid-dino/bin/python \
  5
```

程序每 25 个 batch 以及每个 epoch 边界原子保存一次 epoch 内断点。重新执行 `run_h5.sh` 会从最近的 v2 持久化断点继续。v1 与 v2 的 optimizer-step 语义不同，因此代码会主动拒绝 v1 部分断点；首次运行 v2 前应把旧 H5 输出目录移到其他位置保留。

中断后运行同一命令或 `scripts/resume_h5.sh`。只有 checkpoint fingerprint（检查点指纹）完全一致才会续跑；签名不一致会拒绝复用。训练完成后，可执行：

```bash
bash scripts/package_h5_results.sh \
  /path/to/experiments/beeid/h5-beetrackquery-dev \
  /path/to/h5-analysis-core.tar.gz
```

第一次真实运行后需要人工检查 `h5_method_decision.json`。无论 development 是否通过，本阶段均不得读取 final test；fixed-detector（固定检测器框）仍为 `SERVER_VALIDATION_PENDING`。
