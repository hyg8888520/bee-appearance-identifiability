# H4 冻结开发协议：延迟身份决策与假设隔离记忆

## 研究问题

H1–H3 已表明：更换 frozen backbone 不能稳定消除身份错误，直接对单一 EMA 记忆做可靠性加权也可能形成“错误匹配 → 记忆污染 → 更差匹配”的反馈。H4 不再问“哪个 backbone 更强”，而问两个可证伪问题：

1. 一次逐帧基线 ID switch 出现后，未来 1/3/5/10 帧是否含有足够证据唯一排除错误身份假设？
2. 如果证据存在，局部延迟提交并隔离候选分支的记忆，是否能在不训练 backbone 的情况下减少 IDSW，同时不显著损害 IDF1/HOTA？

所有数值参数、变体、门禁和声明边界均由 `configs/h4_protocol.lock.yaml` 与 SHA-256 sidecar 固定。H3 冻结协议不做任何修改。

## 阶段 A：可恢复性审计

输入是 H4 `immediate_commit` 产生的 development GT-box 轨迹。每个基线 IDSW 构成一个事件；离线 oracle 使用 GT 身份建立事件前历史，并在 `t+[1,3,5,10]` 分别检查：

- appearance：历史 prototype 与未来候选 embedding 的唯一最优一一分配；
- motion：事件前 GT 中心历史外推后的唯一最优一一分配；
- joint：冻结的 0.7 appearance + 0.3 motion；
- 并列一律算不可恢复；未来任一冲突身份缺失也算不可恢复。

GT 只允许写入 `h4_recoverability_events.csv` 等审计产物，字段固定为 `oracle_uses_gt=true`、`method_input=false`。实际 tracker 的任何分数、分支和提交决策都不得读取 `track_id` 或 `identity`。

在 horizon=10 时，ResNet50 与 DINOv3 必须分别满足：

- 至少 10 个基线 IDSW 事件；
- 事件分布在至少 3 个 development 视频；
- joint recoverable fraction ≥ 0.30，分母为全部 IDSW，缺失未来帧不会从分母中删除。

任一 backbone 不满足即输出 `STOP_NO_RECOVERABILITY_SIGNAL`，全量配置禁止继续方法评估。只有 `allow_subset=true` 的 smoke 配置可显式覆盖门禁，以测试程序路径；其结果强制标记 `PROTOCOL_GATE_OVERRIDDEN_FOR_SUBSET_SMOKE_NOT_EXPERIMENT`。

## 阶段 B：事件触发的固定延迟关联

常规帧只运行一次全局 Hungarian。若一个已匹配 track 在局部存在分数差不超过 0.03 的竞争 detection，代码构造相连的二分冲突分量；只处理最多 4 tracks × 4 detections，保留最多 6 个候选分支。过大的冲突不强行截断，而记录 `oversized_conflict` 并退回立即提交。

每个候选分支复制：

- active track 的运动状态；
- track ID 分配器；
- 外观记忆数组；
- 尚未提交的 assignment rows。

分支在 decision frame 之前彼此不可见。窗口结束后只提交累计关联目标最高的分支，其他分支直接丢弃。每一行记录 event start、decision frame 和 latency，因此可以审核任何输出是否读取了 decision frame 之后的数据。

## 冻结消融

- `immediate_commit`：逐帧立即提交和 EMA，H4 对照组。
- `fixed_lag_frozen_memory_h5`：延迟 5 帧，但窗口内所有分支用事件前固定记忆打分；获胜后才回放更新。它分离“多帧/运动证据”与“分支记忆”的贡献。
- `fixed_lag_isolated_memory_h1/h3/h5/h10`：分支内更新独立记忆，分别等待最多 1/3/5/10 帧。
- 主变体固定为 `fixed_lag_isolated_memory_h5`；h10 是 latency 对照，不会事后替换主变体。

## 方法 GO/STOP

对 ResNet50 和 DINOv3 分别比较主变体与 `immediate_commit`：

- IDSW reduction 必须严格大于 0；
- pooled IDF1 与 HOTA 允许的非劣界均为 0.002；
- 6 个 development 视频中至少 4 个同时满足 IDF1/HOTA 不低于基线超过 0.002。

任一 backbone 失败即 `STOP_METHOD_NOT_SUPPORTED`。即使通过，final test 仍保持锁定；必须另建最终协议并重新 checksum，本 H4 开发代码允许的 final-test 运行次数为 0。

## 指标与解释边界

主指标为 AssA、IDF1、IDSW、HOTA；另报告 decision latency、deferred observation fraction、event 数、展开假设数、oversized conflict 数和 video-cluster bootstrap。

GT-box 仍没有 detector FP/FN，不能冒充端到端 MOT。Oracle 可恢复率是机制上限，不是部署性能。H4 也不宣称 multiple-hypothesis tracking、fixed-lag smoothing 或 clip association 本身是新概念。
