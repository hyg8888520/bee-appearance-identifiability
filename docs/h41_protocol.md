# H4.1 冻结探索协议：窗口可恢复性与自适应身份提交

## 研究问题

H4-v1 已证明“在精确 `t+10` 时点仍可恢复”的比例不足，按原协议停止。H4.1 保留这个结论，改问一个不同且更贴近延迟决策的问题：在 IDSW 后的有限窗口中，唯一正确证据是否**曾经出现**；若出现，能否在不读取 GT 身份、不训练 backbone（骨干网络）的情况下及时提交。

所有参数由 `configs/h41_protocol.lock.yaml` 和 SHA-256 sidecar 固定。H4.1 是观察 H4-v1 development 后定义的探索性协议，final test 读取次数为 0。

## 输入与不可变来源

输入包括：

- 冻结 H1 manifest（观测清单）与 project split（项目划分）；
- 完成的 H3 GT-box ResNet50/DINOv3 feature cache（特征缓存），只读复用；
- 完整 H4-v1 STOP 输出目录，用于 provenance（来源）和旧时点逐事件复核；
- 同一 development-validation 视频，不包含 final-test 视频。

启动时要求 H4-v1 `h4_run_metadata.json` 为 `stopped_after_recoverability_audit`，`h4_protocol_decision.json` 为 `STOP_NO_RECOVERABILITY_SIGNAL`。重新计算的 1/3/5/10 帧 availability、appearance/motion/joint recoverability、margin 和 unavailable reason 必须与 H4-v1 一致，否则拒绝运行。

## 阶段 A：连续时点与累计窗口审计

每个 H4-v1 immediate baseline IDSW 是一个事件。离线 oracle（真值诊断器）使用事件前 5 个 GT observation 建立 identity prototype（身份原型）与常速度运动历史，连续检查 `t+1…t+10`。

精确延迟 (h) 的 appearance、motion 和 0.7/0.3 joint 一一分配规则沿用 H4-v1：正确对角分配必须是全局最优，且每行正确项严格高于所有错误项；并列失败。任一冲突身份在该帧不恰好出现一次，则该精确延迟不可用且不可恢复。

对截止时间 (H\in\{1,3,5,10\})：

```text
available_ever(i,H) = OR_{h=1..H} available(i,h)
recoverable_ever(i,H) = OR_{h=1..H} recoverable(i,h)
R_ever(H) = Σ_i recoverable_ever(i,H) / N_all_baseline_IDSW
```

分母始终是全部 baseline IDSW。缺失未来观测不会被删去；代码另外报告 first/last recovery lag、可恢复 lag 数和 availability，避免 survivor bias（幸存者偏差）。

在 H=5 时，ResNet50 与 DINOv3 必须各自满足：事件数至少 10、至少 3 个有事件视频、累计 joint recoverability 至少 30%。任一失败即 `STOP_NO_WINDOW_RECOVERABILITY_SIGNAL`，全量配置禁止运行方法。subset smoke 可测试 STOP/GO 程序路径，但强制标记 `NOT_EXPERIMENT`。

## 阶段 B：自适应延迟提交

冲突触发、局部 component（连通冲突分量）、最大 4×4、beam width（束宽）6、分数和 branch memory（分支记忆）语义继承 H4-v1。候选分支只读取当前帧及历史帧。

每处理一个未来帧，按累计 association objective（关联目标）排序。置信证据定义为：

```text
normalized_margin =
  (best cumulative objective - best competing initial-branch objective)
  / processed_event_frames
```

“竞争分支”必须具有不同的事件起始 assignment；同一起始 assignment 的后续分叉不冒充初始身份冲突。最早 `t+1` 起，若同一起始赢家连续稳定 2 次，且 normalized margin ≥ 0.03，则提前提交；若竞争起始分支已经全部被 beam 淘汰，也视为阈值满足。最迟 `t+5` 强制提交当前最优分支；若视频先结束，则单独记录 `forced_at_sequence_end`，不伪装成达到 H=5。

每个 event/assignment 记录 decision frame、realized horizon、early/forced、normalized margin、稳定次数、使用证据帧数和竞争分支是否仍存活。代码测试保证没有 row 的 frame 超过其 decision frame。

## 冻结消融

- `immediate_commit`：H4-v1 立即提交基线；
- `fixed_lag_frozen_memory_h5`：固定等待 H=5，窗口内记忆冻结；
- `fixed_lag_isolated_memory_h5`：固定等待 H=5，分支记忆隔离；
- `adaptive_lag_frozen_memory_h5`：自适应提前提交，窗口内记忆冻结；
- `adaptive_lag_isolated_memory_h5`：自适应提前提交且分支记忆隔离，冻结主变体。

这个设计分别比较 future evidence（未来证据）、early commitment（提前提交）和 branch-isolated memory（分支隔离记忆），但不把 MHT（多假设跟踪）或提前停止本身宣称为新概念。

## 方法门禁

主变体相对 immediate baseline 对两个 backbone 分别要求：

- IDSW 严格减少；
- pooled IDF1/HOTA 相对下降不超过 0.002；
- 至少 4 个 development 视频同时满足 IDF1/HOTA 非劣。

通过只会输出 `GO_FREEZE_CONFIRMATORY_PROTOCOL`，含义是允许另行冻结确认性/fixed-detector 协议；H4.1 本身仍不能读取 final test。失败输出 `STOP_METHOD_NOT_SUPPORTED`。

## 结果边界

GT 只进入离线 oracle 和事后 metric（指标），绝不进入候选分支、分数、提前提交或记忆更新。GT-box 没有 detector FP/FN，结果不是端到端 MOT。Synthetic encoder（合成测试编码器）只验证程序契约。所有真实 BEE24/RTX 4090 尚未执行项标记 `SERVER_VALIDATION_PENDING`。
