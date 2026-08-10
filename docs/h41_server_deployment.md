# H4.1 服务器运行与交付说明

H4.1 不训练网络，也不重新提取 embedding（特征向量）。它只读扫描 H3 cache，并在 CPU 上运行 oracle 审计、Hungarian（匈牙利匹配）和局部 beam search。因此 GPU 利用率低是正常现象；RTX 4090 已在 H3 特征提取阶段使用，本阶段不是漏用 GPU。

所有真实 BEE24 H4.1 运行在代码交付时均为 `SERVER_VALIDATION_PENDING`。

## 1. 更新代码

```bash
cd ~/projects/bee-appearance-identifiability
git fetch origin
git checkout codex/h41-adaptive-recovery
git pull --ff-only
~/venvs/beeid-dino/bin/python -m pip install --no-deps -e .
```

## 2. 协议与 synthetic smoke

```bash
~/venvs/beeid-dino/bin/python -m beeid.cli h41-validate-protocol \
  --protocol configs/h41_protocol.lock.yaml \
  --checksum configs/h41_protocol.lock.sha256

~/venvs/beeid-dino/bin/python -m beeid.cli h41-synthetic-smoke \
  --output ~/experiments/bee-appearance-identifiability/h41-synthetic
```

Synthetic 输出必须包含：

```text
status: passed
source_h4_v1_status: stopped_after_recoverability_audit
source_h4_v1_conclusion_preserved: true
test_only_encoder: true
real_experiment_result: false
final_test_read: false
```

## 3. 配置路径

```bash
cp configs/h41_smoke.example.yaml configs/h41_smoke.local.yaml
cp configs/h41.example.yaml configs/h41.local.yaml
```

两个 local YAML 都只编辑外部路径：

- `bee24_root`：BEE24 image-frame MOT 树；
- `h1_output_root`：生成冻结 manifest 的完整 H1 输出；
- `h3_output_root`：完整 H3 GT-box 输出，含 cache registry 和 metadata；
- `h4_output_root`：刚才真实 H4-v1 STOP 的**原始完整输出目录**，不能填 H4.1 输出目录，也不能只填下载压缩包；
- `output_root`：全新的 H4.1 smoke/full 目录；
- `cache_root`：保留 H3 cache 根目录；H4.1 不写 feature cache；
- 其余 repo/checkpoint/interpreter 路径沿用 H1–H4，配置类型需要保留，但 H4.1 不加载模型权重。

示例：

```yaml
paths:
  h1_output_root: /root/autodl-tmp/h1
  h3_output_root: /root/autodl-tmp/h3-gt-dev
  h4_output_root: /root/autodl-tmp/h4-deferred-identity-dev
  output_root: /root/autodl-tmp/h41-adaptive-recovery-dev
  cache_root: /root/autodl-tmp/cache-h3
```

## 4. 100–300 帧真实 smoke

```bash
bash scripts/h41_smoke_test.sh \
  configs/h41_smoke.local.yaml \
  ~/venvs/beeid-dino/bin/python
```

subset 可能不满足事件数/视频数门禁，脚本只为覆盖程序路径而允许 `allow_subset=true` override；无论门禁是否恰好通过，metadata 都会标为 `SUBSET_SMOKE_NOT_EXPERIMENT`，不能进入论文表格。

## 5. 全量 development

确认 smoke 后运行：

```bash
bash scripts/run_h41.sh \
  configs/h41.local.yaml \
  ~/venvs/beeid-dino/bin/python
```

顺序固定为：

```text
H1/H3/split/protocol audit
  -> preserved H4-v1 STOP + artifact hash audit
  -> read-only H3 cache scan
  -> recompute immediate baseline
  -> continuous t+1...t+10 oracle audit
  -> exact H4-v1 1/3/5/10 crosscheck
  -> cumulative-by-H recovery gate at H=5
  -> five fixed/adaptive variants (GO only)
  -> paired metrics + video bootstrap + failures + SVG/Markdown guide
```

若累计门禁失败，命令正常结束为 `stopped_after_window_recoverability_audit`；这不是报错，禁止在全量配置使用 override。

## 6. 进度与续跑

```bash
watch -n 5 "cat /root/autodl-tmp/h41-adaptive-recovery-dev/h41_logs/audit_progress.json 2>/dev/null"
watch -n 5 "cat /root/autodl-tmp/h41-adaptive-recovery-dev/h41_logs/tracking_progress.json 2>/dev/null"

bash scripts/resume_h41.sh \
  configs/h41.local.yaml \
  ~/venvs/beeid-dino/bin/python
```

Audit 以 `model×video`、tracking 以 `model×video×variant` 原子保存到 `h41_work/`。Fingerprint 覆盖 observation IDs、H3 cache、H4-v1 event hash、H4.1 protocol 和 gate decision；任何变化都会拒绝旧 job。

## 7. 分阶段排错

```bash
PY=~/venvs/beeid-dino/bin/python
CFG=configs/h41.local.yaml

"$PY" -m beeid.cli h41-validate --config "$CFG"
"$PY" -m beeid.cli h41-audit --config "$CFG" --models resnet50 dinov3
# h41_protocol_decision.json 为 GO_ADAPTIVE_METHOD_EVALUATION 才运行：
"$PY" -m beeid.cli h41-track --config "$CFG" --models resnet50 dinov3
"$PY" -m beeid.cli h41-report --config "$CFG" --models resnet50 dinov3
```

## 8. 打包与发送

```bash
bash scripts/package_h41_results.sh \
  /root/autodl-tmp/h41-adaptive-recovery-dev \
  /root/autodl-tmp/h41-export
```

脚本总会生成 audit 的 core/tables 包；只有 tracker 实际运行时才生成 assignments 包：

- `h41-analysis-core.tar.gz`
- `h41-analysis-tables.tar.gz`
- `h41-assignments.csv.gz`（方法阶段运行后）

打包脚本不会因合法 STOP 缺少 tracking 文件而失败。不要读取 final test；H4.1 metadata 固定显示 `LOCKED_REQUIRES_NEW_CONFIRMATORY_PROTOCOL`。
