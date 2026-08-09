# BEE24 H1 Appearance, H2 Reliability, H2.5 Memory Diagnostics, and Frozen H3 Protocol

## H2.5 refined memory-contamination 与 H3 冻结协议

H2.5 不再使用已被 H2 结果否定的“静态小框或静态低 Laplacian 即低质量”规则。它只在已冻结的 development validation 上，按结果盲的经验百分位选择五类动态风险：身份历史离群、尺度突变、方向代理突变、清晰度变化和拥挤/重叠。对每个事件，代码注入同帧空间最近的不同身份特征，并在完全相同的后续目标上比较四种策略：oracle 正确更新、无条件更新、跳过更新和可靠性加权更新。该实验是 GT 轨迹上的机制测试，不是 MOT 结果。

H3 当前只冻结实验协议，不假装已经实现或运行完整 RAM-Bee tracker。协议锁固定数据隔离、可靠性特征、project-train 阈值拟合、GT/fixed-detector 两阶段、四组消融、最终指标和一次性 final-test 门禁。详见 [H2.5 protocol](docs/h25_protocol.md)、[H3 frozen protocol](docs/h3_frozen_protocol.md) 和 [server deployment](docs/h25_server_deployment.md)。

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
```

真实 `extract --model` 只有三个选项，不包含 synthetic 测试编码器。DINOv3 仅通过 pinned 本地 Hub 的 `dinov3_vits16` 和官方 `forward_features()` 运行，默认使用 `x_norm_patchtokens` mean pooling；没有 timm、DINOv2、Hugging Face 或随机权重 fallback。

## 输出位置

所有结果都写到 YAML 指定、位于仓库外的 `output_root` / `cache_root`：

- `summary.csv`、`per_video_results.csv`、`size_analysis.csv`、`query_results.csv`；
- `failure_cases.csv`、`run_metadata.json`、`resolved_config.yaml`、`resolved_split.yaml`；
- `logs/`、`figures/`；
- cache 在 `<cache_root>/features/<model>/<fingerprint>/`，位置映射记录于 `<output_root>/cache_locations.json`。

H2 另写出 `h2_summary.csv`、observation/query diagnostics、context/paired ablation、clustered and paired bootstrap、factor/predictiveness、memory contamination、`h2_run_metadata.json`、`h2_logs/` 与 `h2_figures/`。完整清单见 [h2_protocol.md](docs/h2_protocol.md)。

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
```

Windows 将 `.venv/bin/python` 替换为 `.venv\Scripts\python.exe`。CPU CI 也只证明单元测试与 synthetic smoke，不声称覆盖 GPU。

## 解释限制

GT-box 实验只衡量 appearance upper bound，不是 detector/tracker 的端到端结果。背景可能泄漏视频、位置或场景信息；扩框会改变背景占比并可能偏向某些模型；Laplacian、方向和 GT-box IoU 只是清晰度、姿态和遮挡代理；development validation 来自 train 视频而非官方 validation。H1/H2 都不能直接证明实际跟踪的 ID Switch 会减少，受控 EMA 污染也不能代替固定检测结果下的端到端 MOT 验证。
