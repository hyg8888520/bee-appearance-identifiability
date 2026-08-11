# BEE24 H1–H5 Appearance Reliability and Identity-State Tracking

## H5：BeeTrackQuery 持久身份查询

H5 从已经冻结的 H3 ResNet50 / DINOv3 crop embedding 出发，只在 `project_train` 训练小型身份关联头，并在 `development_validation` 比较冻结 H3 baseline、无记忆持久查询、无门控短记忆与带可信度门控短记忆。它不微调 backbone，不使用 TOPIC 模型，也不读取 final test。方法设计借鉴 DETR 系跟踪与 DETRAM 的 persistent tracking query（持久跟踪查询）、current-frame refresh（当前帧刷新）和 finite memory（有限记忆）；本项目的重点是蜜蜂同质外观下的 identity-state separation（身份状态分离）与 contamination-aware selective update（防污染选择性更新），不把既有 Transformer 结构本身宣称为创新。

```bash
cp configs/h5.local.yaml.example configs/h5.local.yaml
~/venvs/beeid-dino/bin/python -m beeid.cli h5-validate-protocol \
  --protocol configs/h5_protocol.lock.yaml \
  --checksum configs/h5_protocol.lock.sha256
~/venvs/beeid-dino/bin/python -m beeid.cli h5-synthetic-smoke \
  --output ~/experiments/bee-appearance-identifiability/h5-synthetic
bash scripts/run_h5.sh configs/h5.local.yaml ~/venvs/beeid-dino/bin/python
```

H5 v2 使用带 padding 的 transition batch，每 25 个 batch 原子保存 epoch 内断点，并把只读压缩 H3 shards 转为仓库外 H5 输出目录中的可复用对齐 `.npy`。可在另一个终端运行 `bash scripts/monitor_h5.sh configs/h5.local.yaml ~/venvs/beeid-dino/bin/python 5` 监控。这些是执行优化：backbone、项目划分、训练身份来源和 final-test 门禁仍然冻结；由于 optimizer step 粒度变化，v2 使用新的协议 ID，不能与未完成的 v1 训练混合。

详细定义见 [H5 protocol](docs/h5_protocol.md)、[method/novelty boundary](docs/h5_method_and_novelty.md) 和 [server deployment](docs/h5_server_deployment.md)。真实 RTX 4090、完整 BEE24 和 fixed-detector 验证当前均为 `SERVER_VALIDATION_PENDING`；GT-box 结果不是端到端 MOT。

## H4.1：窗口可恢复性与自适应身份提交

真实 H4-v1 已按冻结协议得到 `STOP_NO_RECOVERABILITY_SIGNAL`：精确 `t+10` joint recoverability 只有 ResNet50 10.60% 和 DINOv3 5.76%。这个结论保持不变。对事件级产物进一步只读复核发现，在已采样 `{1,3,5,10}` 窗口中“至少恢复过一次”的下界分别为 38.41% 和 38.13%，说明证据可能短暂出现或延后出现。H4.1 因而作为**观察 H4-v1 development 后定义的探索性实验**，连续审计 `t+1…t+10`，并测试最迟 H=5、证据稳定即可提前提交、候选记忆隔离的局部关联。

H4.1 启动时必须保留 H4-v1 STOP，且逐事件复核旧 1/3/5/10 时点；任何差异都会拒绝继续。GT 只用于离线 oracle（真值诊断）和事后指标，不进入分支或提交决策。完整依据与边界见 [H4-v1 evidence](docs/h41_h4_v1_evidence.md)、[H4.1 protocol](docs/h41_protocol.md)、[method/novelty](docs/h41_method_and_novelty.md) 与 [server deployment](docs/h41_server_deployment.md)。

本地首次验证：

```bash
~/venvs/beeid-dino/bin/python -m beeid.cli h41-validate-protocol \
  --protocol configs/h41_protocol.lock.yaml \
  --checksum configs/h41_protocol.lock.sha256
~/venvs/beeid-dino/bin/python -m beeid.cli h41-synthetic-smoke \
  --output ~/experiments/bee-appearance-identifiability/h41-synthetic
```

服务器真实运行：

```bash
cp configs/h41_smoke.example.yaml configs/h41_smoke.local.yaml
cp configs/h41.example.yaml configs/h41.local.yaml
# 编辑外部路径，尤其是 h1_output_root、h3_output_root、h4_output_root、output_root。
bash scripts/h41_smoke_test.sh configs/h41_smoke.local.yaml ~/venvs/beeid-dino/bin/python
bash scripts/run_h41.sh configs/h41.local.yaml ~/venvs/beeid-dino/bin/python
```

H4.1 复用 H3 cache，不训练也不重新提取特征，因此主要占用 CPU、GPU 利用率低是预期行为。真实 H4.1、fixed-detector 和 final confirmation 当前均为 `SERVER_VALIDATION_PENDING`；final test 继续锁定。

## H4：延迟身份决策与假设隔离记忆

H4 是独立于已冻结 H3 的新开发阶段。它先审计一次逐帧 IDSW 在未来 1/3/5/10 帧是否可恢复；只有 ResNet50 与 DINOv3 都通过预注册门禁，才运行事件触发的 fixed-lag 局部多假设关联。方法不训练 backbone、不依赖 TOPIC tracker，并通过 `immediate_commit`、frozen-memory h5 及 isolated-memory h1/h3/h5/h10 六个变体区分未来证据、延迟和分支记忆隔离的作用。

H4 只读复用完整 H3 cache，所以服务器运行主要占用 CPU，GPU 利用率低是预期现象。完整定义见 [H4 protocol](docs/h4_protocol.md)、[method/novelty boundary](docs/h4_method_and_novelty.md) 与 [server deployment](docs/h4_server_deployment.md)。首次运行：

```bash
cp configs/h4.local.yaml.example configs/h4.local.yaml
~/venvs/beeid-dino/bin/python -m beeid.cli h4-validate-protocol \
  --protocol configs/h4_protocol.lock.yaml \
  --checksum configs/h4_protocol.lock.sha256
~/venvs/beeid-dino/bin/python -m beeid.cli h4-synthetic-smoke \
  --output ~/experiments/bee-appearance-identifiability/h4-synthetic
bash scripts/h4_smoke_test.sh configs/h4_smoke.local.yaml ~/venvs/beeid-dino/bin/python
bash scripts/run_h4.sh configs/h4.local.yaml ~/venvs/beeid-dino/bin/python
```

真实 BEE24 H4-v1 development 审计已经完成并按协议停止；方法阶段没有运行。synthetic feature 只验证程序契约，不是实验结果。H4 development 协议允许的 final-test 读取次数为 0。

## H2.5 refined memory-contamination 与 H3 冻结协议

H2.5 不再使用已被 H2 结果否定的“静态小框或静态低 Laplacian 即低质量”规则。它只在已冻结的 development validation 上，按结果盲的经验百分位选择五类动态风险：身份历史离群、尺度突变、方向代理突变、清晰度变化和拥挤/重叠。对每个事件，代码注入同帧空间最近的不同身份特征，并在完全相同的后续目标上比较四种策略：oracle 正确更新、无条件更新、跳过更新和可靠性加权更新。该实验是 GT 轨迹上的机制测试，不是 MOT 结果。

H3 已实现冻结协议中的第一阶段 `gt_detection_boxes`：project-train-only 可靠性校准、因果在线关联、四组消融、GT-box identity metrics、视频聚类 bootstrap 和 final-test 门禁。`fixed_detector_boxes` 仍为 `SERVER_VALIDATION_PENDING`，在 GT-box development 通过前不会读取 final test。详见 [H2.5 protocol](docs/h25_protocol.md)、[H3 frozen protocol](docs/h3_frozen_protocol.md)、[H3 implementation](docs/h3_implementation.md) 和 [H3 server deployment](docs/h3_server_deployment.md)。

服务器上先从已完成的 H2 目录复用特征，无需再次提取：

```bash
cd ~/projects/bee-appearance-identifiability
cp configs/h25.example.yaml configs/h25.local.yaml
# 只编辑外部路径；尤其是 h1_output_root、h2_output_root、output_root 和 cache_root。
~/venvs/beeid-dino/bin/python -m beeid.cli h3-validate-protocol \
  --protocol configs/h3_protocol.lock.yaml \
  --checksum configs/h3_protocol.lock.sha256
~/venvs/beeid-dino/bin/python -m beeid.cli h25-synthetic-smoke \
  --output ~/experiments/bee-appearance-identifiability/h25-synthetic
bash scripts/run_h25.sh configs/h25.local.yaml ~/venvs/beeid-dino/bin/python
```

H2.5 输出包括 `h25_events.csv`、`h25_strategy_trajectories.csv`、`h25_strategy_summary.csv`、`h25_paired_summary.csv`、`h25_cluster_bootstrap.csv`、阈值/metadata、日志和 PNG figures。H2.5 的真实 BEE24、真实 cache 和 RTX 4090 运行在提交代码时仍为 `SERVER_VALIDATION_PENDING`；synthetic encoder 绝不作为实验模型或结果。

本仓库实现 BEE24 的 GT-box 外观可辨识性上限基准 H1，以及 observation reliability 与轨迹模板污染诊断 H2。H1 比较 ImageNet ResNet50、DINOv3 ViT-S/16 与 TOPICTrack 官方 BEE AGW；H2 研究时间、清晰度、重叠、邻近密度、边界、上下文和历史一致性何时会使这些外观特征不可靠。

本项目不是 DINOv3、TOPICTrack、FastReID 或 BEE24 的官方实现；它只通过固定版本的官方接口组织独立 H1/H2 实验。

> 当前状态：首次完整 H1 已在 RTX 4090、完整 BEE24 和真实 checkpoint 上运行并完成结果审计，固定结论见 [h1_findings.md](docs/h1_findings.md)。H2 源码、CPU 单元测试和 test-only synthetic smoke 可在本地验证；真实 H2 多干预特征、可靠性统计和模板污染结果仍为 `SERVER_VALIDATION_PENDING`。两阶段都只使用 development validation，最终 test 继续锁定。

H3 源码的 CPU 单元测试与 test-only synthetic smoke 可在本地验证；真实 project-train 特征提取、阈值拟合、GT-box development 和 fixed-detector development 在代码提交时统一标记 `SERVER_VALIDATION_PENDING`。synthetic encoder 只测试程序契约，绝不进入真实模型选项或实验结果。

## 许可与使用边界

本项目原创代码**未明确授予任何许可**，仓库没有顶层 `LICENSE`。查看或取得源码不等于获得复制、修改或再分发许可。各上游项目仍受其各自许可证约束，详见 [reference_audit.md](docs/reference_audit.md) 与 [references.lock.yaml](references.lock.yaml)。BEE24 官方页面未说明数据许可证，本项目将其标记为 `LICENSE_NOT_STATED_BY_SOURCE`，仅处理使用者合法取得的数据，绝不提交或再分发。

## 方法概览

- 从 `train|test/<video>/seqinfo.ini`、`img1/`、`gt/gt.txt` 读取 MOT 数据；身份定义为 `(video_id, track_id)`。缺失 `seqinfo.ini` 默认报错，也可显式设置 `missing_seqinfo_policy: infer_from_images`，从经过帧名、扩展名、连续性和尺寸验证的图片目录推断，并生成审计记录。
- one-based MOT 坐标转换为 zero-based 半开区间；bbox 中心扩展 20%（每侧 10%），再 `floor/ceil` 并裁边；crop 按需读取，不落盘数十万图片。
- train 视频按 `SHA256("24:<video_id>")` 排序，选取 `max(1, round(20% × n))` 作为项目 validation；test 不参与选择。
- 查询帧 `t` 对候选帧 `t+[1,5,10,25]`。Full gallery 包含候选帧所有有效目标；Hard gallery 是唯一 positive 加空间最近的最多 5 个不同身份。
- cosine 并列一律算 Rank-1 失败；无 positive 不进分母，无 negative 不进 Hard Rank-1 分母，但都记录原因。
- 所有 cache 指纹覆盖 manifest、模型/上游版本、checkpoint SHA-256、transform、尺寸和 AMP；分片先临时写入再原子重命名，可安全续跑。

完整定义见 [research_protocol.md](docs/research_protocol.md)。项目 train/development-validation/final-test 的冻结清单见 [project_split.yaml](configs/splits/project_split.yaml)，官方 AGW 的解释边界见 [agw_training_overlap_audit.md](docs/agw_training_overlap_audit.md)。

## H2：观测可靠性与模板污染

H2 不重新划分数据，也不覆盖 H1。它从已完成的 H1 manifest 读取同一批 development observations，并对每个 backbone 运行固定干预：0/20/50% 上下文、224/256 输入、矩形 bbox 前景/背景对照和 Gaussian blur。结果盲信号包括 bbox/patch coverage、Laplacian 清晰度、结构方向代理、GT-box 重叠、邻近密度、边界、轨迹年龄和历史 prototype 一致性。

统计同时报告样本量、Rank-1、margin、逐视频结果，以及 video/identity 聚类 bootstrap；不会把相邻帧当成独立样本。受控模板实验在 GT 轨迹上向普通 EMA 注入低质量 observation、模糊特征或同帧错误身份，测量相似度损失、诱发错误和恢复步数。详细、预先固定的定义见 [h2_protocol.md](docs/h2_protocol.md)。这一步只诊断机制，不提前实现 RAM-Bee，也不等价于完整 MOT。

## H3：可靠性感知关联

H3 不复用 development Q75 作为方法参数。代码从冻结 H1 manifest 选择 25 个 `project_train` 视频和 6 个 `development_validation` 视频，对两部分生成统一 crop embedding 与 outcome-blind signals；所有 CDF/阈值只在 `project_train` 拟合，并将 manifest、协议、split、signal 和 cache 指纹写入 `h3_thresholds.json`。

四组冻结变体是：

- `baseline_association`：外观与常速度运动关联，无条件 EMA；
- `selective_memory_update`：相同关联，低可靠 observation 跳过记忆更新；
- `reliability_weighted_association`：可靠性降低 appearance association 权重，记忆仍无条件更新；
- `full_ram_bee`：可靠性感知关联加 hard gate 和 weighted EMA。

GT-box 阶段直接知道每个 tracker detection 对应哪一个 GT detection，因此 DetA=1、Frag=0 是阶段设计的结果；这里的 HOTA 只反映 association。IDF1、AssA、HOTA 与 IDSW 的语义对照 pinned TrackEval `12c8791b`，但不把 GT-box 结果冒充固定 detector 或完整检测跟踪结果。

## 服务器首次部署

以下命令均在 Linux 服务器执行，不需要安装 Codex。建议目录仅作示例，业务代码不会硬编码它们。

```bash
git clone https://github.com/hyg8888520/bee-appearance-identifiability.git ~/projects/bee-appearance-identifiability
git clone https://github.com/facebookresearch/dinov3.git ~/third_party/dinov3
git -C ~/third_party/dinov3 checkout 6876159a11b4df116f30f667f8c9888617df0751
git clone https://github.com/holmescao/TOPICTrack.git ~/third_party/TOPICTrack
git -C ~/third_party/TOPICTrack checkout e7b260f41a92fbe94419ce448aa7b53fe80a6814

cd ~/projects/bee-appearance-identifiability
bash scripts/setup_dino_env.sh ~/venvs/beeid-dino
bash scripts/setup_topic_env.sh ~/venvs/beeid-topic
cp configs/h1.local.yaml.example configs/h1.local.yaml
cp configs/h1_smoke.example.yaml configs/h1_smoke.local.yaml
cp configs/h2.local.yaml.example configs/h2.local.yaml
cp configs/h2_smoke.example.yaml configs/h2_smoke.local.yaml
cp configs/h3.local.yaml.example configs/h3.local.yaml
cp configs/h3_smoke.example.yaml configs/h3_smoke.local.yaml
# 按所运行阶段编辑 *.local.yaml 中的全部路径；这些文件已被 Git 忽略。
```

两个环境都使用 Python 3.11、官方 `torch==2.7.1` / `torchvision==0.22.1` CUDA 12.8 wheel。驱动报告 CUDA 13.0 并不要求安装 CUDA 13.0 PyTorch；不要修改/降级 NVIDIA driver，也不默认源码编译 PyTorch。TOPIC 环境只安装 FastReID 推理所需的最小现代依赖，不安装旧 PyTorch 1.8、YOLOX、detector 或 tracker。

## 固定验证顺序

先激活 DINO 环境（编排器会从 YAML 调用两个环境的解释器）：

```bash
# 1. 分别核对两个现代环境的路径、commit、PyTorch/CUDA/设备
source ~/venvs/beeid-dino/bin/activate
bash scripts/check_server_env.sh configs/h1_smoke.local.yaml
~/venvs/beeid-topic/bin/python -m beeid.cli server-check --config configs/h1_smoke.local.yaml

# 2. 在包含 TOPIC 最小依赖的现代环境中完成全部真实 crop GPU 检查
~/venvs/beeid-topic/bin/python scripts/gpu_smoke.py --config configs/h1_smoke.local.yaml

# 3. synthetic smoke + 配置所限的 100–300 帧真实 smoke
bash scripts/smoke_test.sh configs/h1_smoke.local.yaml
```

GPU 报告写入 `<output_root>/logs/gpu_smoke.json`。若 AGW 无法在现代环境严格加载，报告 `BLOCKED_TOPIC_AGW_COMPATIBILITY`，但 ResNet50 与 DINOv3 流程仍可继续；严禁用普通 ResNet50 冒充 AGW。

真实 smoke 通过后：

1. 在 local YAML 设置一个 `dataset.video_ids`，清空 `max_frames_per_video`，运行完整单视频。
2. 用 `beeid estimate` 填入实测速率、embedding 维度和峰值显存，保存时间/磁盘估算。
3. 由用户确认后，才运行 `bash scripts/run_h1.sh configs/h1.local.yaml`。全量命令在代码中受 `--confirm-full` 门禁保护。
4. 中断后运行 `bash scripts/resume_h1.sh configs/h1.local.yaml`；只有签名和 observation IDs 完全一致的合法分片会复用。

Scheduler 尚未确认，因此仓库不含 Slurm 脚本。

## 可组合 CLI

```bash
beeid validate-data --config configs/h1.local.yaml
beeid build-manifest --config configs/h1.local.yaml
beeid extract --config configs/h1.local.yaml --model resnet50
beeid extract --config configs/h1.local.yaml --model dinov3
beeid extract --config configs/h1.local.yaml --model topic_agw
beeid evaluate --config configs/h1.local.yaml --models resnet50 dinov3 topic_agw
beeid report --config configs/h1.local.yaml --models resnet50 dinov3 topic_agw
beeid estimate --config configs/h1.local.yaml --embedding-dimension 384 --observations-per-second 100
beeid synthetic-smoke

# H2 requires a completed H1 output root and a separate H2 output root.
beeid h2-validate --config configs/h2.local.yaml
beeid h2-signals --config configs/h2.local.yaml
beeid h2-extract --config configs/h2.local.yaml --model resnet50
beeid h2-extract --config configs/h2.local.yaml --model dinov3
beeid h2-extract --config configs/h2.local.yaml --model topic_agw
beeid h2-evaluate --config configs/h2.local.yaml --models resnet50 dinov3 topic_agw
beeid h2-contamination --config configs/h2.local.yaml --models resnet50 dinov3 topic_agw
beeid h2-report --config configs/h2.local.yaml --models resnet50 dinov3 topic_agw
beeid h2-estimate --config configs/h2.local.yaml --model dinov3 --embedding-dimension 384 --observations-per-second 100
beeid h2-synthetic-smoke

# H3 uses H1's frozen manifest but writes independent train/dev caches and outputs.
beeid h3-validate-protocol --protocol configs/h3_protocol.lock.yaml --checksum configs/h3_protocol.lock.sha256
beeid h3-validate --config configs/h3.local.yaml
beeid h3-signals --config configs/h3.local.yaml
beeid h3-extract --config configs/h3.local.yaml --model resnet50
beeid h3-extract --config configs/h3.local.yaml --model dinov3
beeid h3-fit-thresholds --config configs/h3.local.yaml --models resnet50 dinov3
beeid h3-track --config configs/h3.local.yaml --models resnet50 dinov3
beeid h3-report --config configs/h3.local.yaml --models resnet50 dinov3
beeid h3-synthetic-smoke
```

真实 `extract --model` 只有三个选项，不包含 synthetic 测试编码器。DINOv3 仅通过 pinned 本地 Hub 的 `dinov3_vits16` 和官方 `forward_features()` 运行，默认使用 `x_norm_patchtokens` mean pooling；没有 timm、DINOv2、Hugging Face 或随机权重 fallback。

## 输出位置

所有结果都写到 YAML 指定、位于仓库外的 `output_root` / `cache_root`：

- `summary.csv`、`per_video_results.csv`、`size_analysis.csv`、`query_results.csv`；
- `failure_cases.csv`、`run_metadata.json`、`resolved_config.yaml`、`resolved_split.yaml`；
- `logs/`、`figures/`；
- cache 在 `<cache_root>/features/<model>/<fingerprint>/`，位置映射记录于 `<output_root>/cache_locations.json`。

H2 另写出 `h2_summary.csv`、observation/query diagnostics、context/paired ablation、clustered and paired bootstrap、factor/predictiveness、memory contamination、`h2_run_metadata.json`、`h2_logs/` 与 `h2_figures/`。完整清单见 [h2_protocol.md](docs/h2_protocol.md)。

H3 写出 `h3_observation_signals.csv`、`h3_thresholds.json`、`h3_assignments.csv`、`h3_per_video_metrics.csv`、`h3_summary.csv`、`h3_paired_video_metrics.csv`、`h3_video_cluster_bootstrap.csv`、MOTChallenge tracker text、metadata、resolved config 和日志。特征 cache 在 `<cache_root>/features/h3__<model>/<fingerprint>/`。

不要把这些目录指回仓库。`configs/*.local.yaml`、权重、`.npz` cache、数据和常见输出目录均被 `.gitignore` 排除。

## 本地 CPU 验证

使用独立环境，不修改 Codex 捆绑运行时：

```bash
python -m venv .venv
.venv/bin/python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest -q
.venv/bin/beeid synthetic-smoke
.venv/bin/beeid h2-synthetic-smoke
.venv/bin/beeid h25-synthetic-smoke
.venv/bin/beeid h3-synthetic-smoke
```

Windows 将 `.venv/bin/python` 替换为 `.venv\Scripts\python.exe`。CPU CI 也只证明单元测试与 synthetic smoke，不声称覆盖 GPU。

## 解释限制

GT-box H1 只衡量 appearance upper bound；H3 GT-box 虽然产生在线关联轨迹，仍没有 detector FP/FN，不能当成完整 MOT。背景可能泄漏视频、位置或场景信息；扩框会改变背景占比并可能偏向某些模型；Laplacian、方向和 GT-box IoU 只是清晰度、姿态和遮挡代理；development validation 来自 train 视频而非官方 validation。只有 fixed-detector development 方向稳定、全部 freeze gate 完成后，才允许一次 final-test evaluation。
