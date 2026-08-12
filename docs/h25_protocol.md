# H2.5 refined memory-contamination protocol

## 研究问题

H2 已显示：动态的尺度、方向、清晰度和身份历史变化能预测 appearance failure，而静态 bbox area 与静态 Laplacian variance 在视频内几乎没有稳定预测力；原 `observed_low_quality` 合并规则也没有在平均意义上造成模板退化。因此 H2.5 不再问“某个静态低质量阈值是否污染模板”，而问：当动态风险时刻发生身份污染时，跳过或降低更新权重能否比无条件更新更稳健？

机器可读定义以 [`configs/h25_protocol.lock.yaml`](../configs/h25_protocol.lock.yaml) 为唯一参数来源。YAML local config 只提供路径、已冻结 H2 variant 定义和 subset/full 运行范围，不能静默修改 H2.5 参数。

## 输入与隔离

- 只读已完成 H1 manifest 与 H2 `context_e020_raw_i224` primary feature cache。
- 只使用冻结的 `development_validation`；`final_test_access` 固定为 `false`。
- 身份为 `(video_id, track_id)`，事件与恢复均沿 GT identity trajectory 构造。
- TOPIC 官方 BEE AGW 仅为 `REFERENCE_ONLY_TRAINING_OVERLAP_UNCONFIRMED`。
- H2 metadata、signal CSV、cache registry、manifest 和两个协议锁的 SHA-256 写入 H2.5 metadata。

## 结果盲事件

每条轨迹首先寻找 5 个连续帧 observation 作为 trusted history。之后仅用当前及过去信息计算：

1. `history_outlier`：当前 embedding 与前 5 个同身份 embedding prototype 的 `1-cosine`；
2. `bbox_scale_change`：相邻 observation bbox area 的绝对 log ratio；
3. `orientation_change_proxy`：相邻 bbox structure-tensor orientation 的 180°轴向差；
4. `sharpness_change`：相邻 bbox Laplacian variance 的绝对 log ratio；
5. `crowding`：同帧 wide-neighbor count 与 max GT-box IoU 的经验百分位最大值。

静态 bbox area 与静态 absolute Laplacian 明确排除。选择过程不读取 Rank-1、margin、相似度标签或未来 observation。每个模型、身份、事件类型最多选择第一个达到 development empirical Q75、具有同帧不同身份候选和后续 observation 的时刻。`wrong_identity_control` 是无风险阈值的正控制，用于检查注入机制是否生效。

这些 development 百分位只用于 H2.5 机制诊断。H3 可部署阈值必须重新在 `project_train` 拟合，不能把 development empirical distribution 带入 final test。

## 四种严格配对策略

事件 contaminant 固定为同帧中心距离最近的不同身份 feature。基础 EMA `alpha=0.2`：

- `oracle_correct_update`：用当前正确身份 feature 完整更新；
- `unconditional_update`：用 contaminant 完整更新；
- `skip_update`：事件时不更新；
- `reliability_weighted_update`：用 contaminant 更新，但 alpha 乘以 `clip(1-max_risk_percentile, 0.05, 1)`。

事件后所有策略都用相同的正确身份 future observations 以基础 alpha 更新，最长 10 步。每一步使用同一帧完整 gallery 计算 Rank-1；与最佳 negative 并列算失败。报告相对 oracle 的 similarity gap、induced error 和恢复步，以及 skip/weighted 相对 unconditional 的精确配对差值。

## 统计与输出

固定报告 immediate（step 1）、short（1–3）和 horizon（1–10）窗口。对 rank1 gain、similarity-gap reduction 和 induced-error reduction 分别按 video 与 identity cluster 进行 1,000 次 seed-24 bootstrap。少量视频下区间是描述性证据，不作强因果或普适性声明。

输出位于外部 `output_root`：

- `h25_event_thresholds.json`
- `h25_events.csv`
- `h25_strategy_trajectories.csv`
- `h25_strategy_summary.csv`
- `h25_paired_summary.csv`
- `h25_cluster_bootstrap.csv`
- `h25_experiment_metadata.json`
- `h25_run_metadata.json`
- `h25_resolved_config.yaml`
- `h25_logs/` 与 `h25_figures/`

H2.5 只能支持“可靠性控制具有机制收益/无收益”的开发集结论。它不能直接支持 IDSW、IDF1、HOTA 或端到端 tracking 改进声明。
