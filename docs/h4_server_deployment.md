# H4 服务器运行说明

H4 不训练网络，也不重新提取 embedding。它只读复用已经完成的 H3 ResNet50/DINOv3 cache；主要计算是 CPU 上的 NumPy/Hungarian 和局部假设枚举，所以 GPU 利用率很低是预期行为，不代表代码偷偷改用 CPU 训练。RTX 4090 只在此前 H3 feature extraction 时发挥作用。

## 1. 更新代码与环境

```bash
cd ~/projects/bee-appearance-identifiability
git fetch origin
git checkout codex/h4-deferred-identity
git pull --ff-only
~/venvs/beeid-dino/bin/python -m pip install --no-deps -e .
```

## 2. 校验协议与 synthetic smoke

```bash
~/venvs/beeid-dino/bin/python -m beeid.cli h4-validate-protocol \
  --protocol configs/h4_protocol.lock.yaml \
  --checksum configs/h4_protocol.lock.sha256

~/venvs/beeid-dino/bin/python -m beeid.cli h4-synthetic-smoke \
  --output ~/experiments/bee-appearance-identifiability/h4-synthetic
```

输出必须显示 `test_only_encoder: true`、`real_experiment_result: false` 和 `final_test_read: false`。

## 3. 100–300 帧真实 smoke

```bash
cp configs/h4_smoke.example.yaml configs/h4_smoke.local.yaml
```

必须填写：

- `bee24_root`：BEE24 根目录；
- `h1_output_root`：产生冻结 manifest 的完整 H1 输出；
- `h3_output_root`：已显示 `completed_development_gt_boxes` 的 H3 输出；
- `output_root`：新的 H4 smoke 目录；
- `cache_root`：保留原 H3 cache 位置即可，H4 不写新 feature cache；
- 其余 checkpoint/repo 路径由通用配置类型要求保留，H4 本身不会加载它们。

然后运行：

```bash
bash scripts/h4_smoke_test.sh \
  configs/h4_smoke.local.yaml \
  ~/venvs/beeid-dino/bin/python
```

300 帧可能没有 10 个 IDSW 或 3 个有事件的视频，因此脚本只在 `allow_subset=true` 时覆盖 oracle gate 来测试 tracker 路径；所有结果会标为 `NOT_EXPERIMENT`，不能纳入论文表格。

## 4. 全量 development

```bash
cp configs/h4.local.yaml.example configs/h4.local.yaml
# 填写外部路径；保持 allow_subset=false，禁止添加 video_ids/max_frames。
bash scripts/run_h4.sh \
  configs/h4.local.yaml \
  ~/venvs/beeid-dino/bin/python
```

流程固定为：

```text
H1/H3/split/checksum audit
  -> read-only H3 cache scan
  -> immediate baseline + 1/3/5/10-frame oracle audit
  -> recoverability GO/STOP
  -> six frozen association variants (GO only)
  -> metrics + paired video bootstrap + failure audit + SVG
```

若 oracle gate 失败，`h4-run-all` 会正常结束并报告 `stopped_after_recoverability_audit`；这不是程序错误，也不应使用全量 override。中断后可运行：

```bash
bash scripts/resume_h4.sh \
  configs/h4.local.yaml \
  ~/venvs/beeid-dino/bin/python
```

进度文件：

```bash
watch -n 5 "cat /path/to/h4-output/h4_logs/audit_progress.json 2>/dev/null"
watch -n 5 "cat /path/to/h4-output/h4_logs/tracking_progress.json 2>/dev/null"
```

每个 audit job 按 `model×video`、每个 tracking job 按 `model×video×variant` 原子写入 `<output_root>/h4_work/`。重新运行时只有 fingerprint 完全一致的 job 才会复用；协议、H3 cache、observation IDs 或 oracle decision 任一变化都会自动拒绝旧 checkpoint。

## 5. 分阶段命令

```bash
PY=~/venvs/beeid-dino/bin/python
CFG=configs/h4.local.yaml

"$PY" -m beeid.cli h4-validate --config "$CFG"
"$PY" -m beeid.cli h4-audit --config "$CFG" --models resnet50 dinov3
# 只有 h4_protocol_decision.json 为 GO_METHOD_EVALUATION 才执行：
"$PY" -m beeid.cli h4-track --config "$CFG" --models resnet50 dinov3
"$PY" -m beeid.cli h4-report --config "$CFG" --models resnet50 dinov3
```

## 6. 结果与打包

核心输出：

- `h4_protocol_decision.json`：是否允许进入方法评估；
- `h4_recoverability_events.csv` / `h4_horizon_summary.csv`：GT oracle 审计；
- `h4_assignments.csv` / `h4_conflict_events.csv`：因果 tracker 输出；
- `h4_summary.csv` / `h4_per_video_metrics.csv` / `h4_paired_video_metrics.csv`；
- `h4_video_cluster_bootstrap.csv` / `h4_failure_cases.csv`；
- `h4_method_decision.json` / `h4_run_metadata.json`；
- `h4_runtime_profile.csv`、`h4_figures/`、`h4_mot_results/`、`h4_logs/`。

不要手工输入带尾随空格的反斜杠 tar 命令。使用仓库脚本：

```bash
bash scripts/package_h4_results.sh \
  ~/experiments/bee-appearance-identifiability/h4-deferred-identity-dev \
  ~/autodl-tmp/h4-export
```

发送生成的三个文件：`h4-analysis-core.tar.gz`、`h4-analysis-tables.tar.gz`、`h4-assignments.csv.gz`。不要运行 final test；H4 metadata 会继续显示 `LOCKED_REQUIRES_NEW_PROTOCOL`。
