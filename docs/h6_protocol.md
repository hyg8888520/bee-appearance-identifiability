# H6：全局时空轨迹推理与可学习选择

H6 不再尝试给 H3/TOPIC 增加一个小型记忆模块，也不以“DINOv3 胜过 ResNet50”为目标。它检验更强的命题：蜜蜂外观高度同质时，是否需要对一段时间内的所有观测做联合身份推理，并学习何时采用这种推理、何时完整保留基线。

## 方法

`GlobalTrajectoryReasoner`（全局轨迹推理器）接收冻结 H3 crop embedding、归一化框几何和帧位置。每个窗口先聚合 frame token（帧级群体上下文），再用双向 Transformer 在 64 帧窗口上推理。pair head（观测对关联头）给所有时间差不超过 12 帧的跨帧观测对打分；推理阶段不存在固定 top-K 候选剪枝。

所有高置信边按分数进入一个带同帧互斥约束的全局图聚类器。它输出 `global_trajectory_reasoner`。另一个可训练 `TrajectorySelector`（轨迹选择器）比较神经分区和冻结 H3 baseline 分区的重叠连通分量；选择单元是完整分量，不能只接受一次拆分或合并的一半。

选择器只用以下可观测量：轨迹长度/覆盖率、边概率及 margin（与竞争边的差值）、空间步长、涉及的基线/神经轨迹数和分区分歧率。GT identity（真实身份）仅在 project-train 离线生成“相对基线的 pair-partition 净收益”标签，并且不进入模型决策。

## 划分和失败闭环

- `project_train_fit`：训练关联网络和选择器。
- `project_train_calibration`：按 `SHA256(seed=24, video_id)` 隔离出的 20% 视频；只用于选择风险阈值。
- `development_validation`：阈值冻结后才评估。
- `final_test`：开发期读取次数为 0。

关联网络训练后，先在 calibration 视频上按 pair precision/recall（观测对精确率/召回率）自动选择图边阈值；配置中的 `min_assignment_probability=0.1` 只是数值退化保护和搜索下界，不是最终人工阈值。随后选择器阈值必须同时满足 precision、harm rate、事件数和视频数约束。找不到安全阈值时，primary variant 的每个分量都逐观测复现冻结 baseline，并报告 `STOP_NO_SAFE_SELECTOR_THRESHOLD`；不人为调阈值。

## 训练目标与计算

关联网络同时优化 balanced BCE（平衡二元关联损失）、supervised contrastive loss（监督对比损失）和 path cycle consistency（路径循环一致性）。训练和推理调用完全相同的 `encode()`/`pair_logits()` 路径。窗口以总 token 数组成 GPU batch，pair head 分块执行，适配 RTX 4090 AMP。

checkpoint 原子写入并锁定协议、H3 cache、观测集合、网络规格和分区，同时保存 next batch、optimizer、GradScaler 及 CPU/CUDA RNG 状态。签名不一致拒绝复用。

## 思路来源

- [MOTIP（CVPR 2025）](https://openaccess.thecvf.com/content/CVPR2025/html/Gao_Multiple_Object_Tracking_as_ID_Prediction_CVPR_2025_paper.html)：把多目标跟踪表述为身份预测，启发“直接学习身份关联”而不是给固定匹配器调权重。
- [Global Tracking Transformers（CVPR 2022）](https://openaccess.thecvf.com/content/CVPR2022/html/Zhou_Global_Tracking_Transformers_CVPR_2022_paper.html)：启发在长窗口内做全局目标交互。
- [3DMOTFormer（ICCV 2023）](https://openaccess.thecvf.com/content/ICCV2023/html/Ding_3DMOTFormer_Graph_Transformer_for_Online_3D_Multi-Object_Tracking_ICCV_2023_paper.html)：启发用图 Transformer 联合建模时空关系。
- [Path Consistency（CVPR 2024）](https://openaccess.thecvf.com/content/CVPR2024/html/Lu_Self-Supervised_Multi-Object_Tracking_with_Path_Consistency_CVPR_2024_paper.html)：启发路径循环一致性辅助目标。
- [DeconfuseTrack（CVPR 2024）](https://openaccess.thecvf.com/content/CVPR2024/html/Huang_DeconfuseTrack_Dealing_with_Confusion_for_Multi-Object_Tracking_CVPR_2024_paper.html)：启发显式处理混淆身份而非只增强 backbone。
- [Multiple Hypothesis Tracking Revisited（ICCV 2015）](https://openaccess.thecvf.com/content_iccv_2015/html/Kim_Multiple_Hypothesis_Tracking_ICCV_2015_paper.html)：启发保留并比较轨迹级替代解释。

这些工作提供方法论来源；H6 的“基线/神经分区重叠分量 + 学习选择 + 精确回退”是针对本项目 SLTR/H5 失败证据形成的实验设计，不宣称已经获得真实 BEE24 改进。真实 RTX 4090、完整数据、固定 detector 的验证均为 `SERVER_VALIDATION_PENDING`。
