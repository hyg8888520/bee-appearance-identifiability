# H5 BeeTrackQuery 冻结实验协议

H5 的问题不是“DINOv3 是否比 ResNet50 强”，而是：在冻结外观 backbone（骨干特征网络）后，显式的 identity state（身份状态）与受控更新，能否比逐帧即时关联更可靠地处理 BEE24 的同质外观目标。

## 数据边界

- `project_train`：唯一允许训练 BeeTrackQuery 参数的分区。
- `development_validation`：只用于消融、门控决策与误差分析。
- `final_test`：H5 development 阶段读取次数为 0。
- 身份始终为 `(video_id, track_id)`；视频是隔离单位，状态绝不跨视频。
- 输入是 H3 已完成的只读 frozen embedding（冻结特征），不重新提取，不微调 ResNet50 或 DINOv3。

## 方法与消融

BeeTrackQuery 将每个活动轨迹表示成 persistent track query（持久跟踪查询）。查询用当前帧 detection tokens（检测框特征 token）做 cross-attention refresh（交叉注意力刷新），再从最近 4 个可靠状态构成的短记忆读取 top-2 内容。association head（关联头）输出轨迹—检测匹配分数；reliability head（可信度头）在 `project_train` 上用“当前最高分匹配是否正确”作二元监督。

冻结比较为：

1. `frozen_h3_baseline`：原 H3 的即时 appearance+motion（外观加运动）关联，按原结果只读复用。
2. `persistent_query_no_memory`：持久查询，但无历史记忆。
3. `beetrackquery_short_memory`：短记忆，每次匹配都更新。
4. `beetrackquery_gated_memory`：短记忆，可信度低于 0.70 时冻结身份记忆；这是 primary variant（主方法）。

主指标是 AssA（关联准确率）、IDF1（身份 F1）、IDSW（身份切换）与 HOTA。在 GT-box 阶段 DetA=1 是构造结果，不能解释成检测器性能。可信度另报 Brier score（概率均方误差）和 ECE（期望校准误差）。

## 训练执行约束

监督样本仍由长度为 4、步长为 2 的连续帧 clip（片段）产生，但实现会把其中的相邻帧 transition（转移样本）展平，再做带 padding（补齐）的批量训练。`runtime.batch_size` 限制 transition 数，`max_pair_elements_per_batch=262144` 同时限制批内 `batch × query × detection` 的上界，以免拥挤帧造成显存峰值。该优化不改变数据分区、身份标签来源、冻结 backbone 或 final-test 门禁，但会改变 optimizer step（优化器更新）的粒度，因此实现指纹独立标为 `beeid.h5:v2-batched-transitions`，不得续用 v1 的部分训练断点。

压缩 H3 shards（分片）只在首次运行时严格核验并扫描一次；按 H5 observation 顺序生成的连续 `.npy` 只写入仓库外 `output_root/h5_embedding_cache/`。后续训练与跟踪按 fingerprint（指纹）复用该文件，不重新提取特征。训练每 25 个 batch 和每个 epoch（轮次）边界原子保存断点，每个 batch 更新可监控的进度 JSON。

## development gate

两个 backbone 都必须满足：primary 的 IDF1 与 HOTA 相对 baseline 不劣超过 0.002；IDSW 严格减少；至少 4 个 development 视频不受损。未通过则输出 `STOP_OR_REVISE_BEETRACKQUERY`，不解锁 final test。

完整机器可读协议与 SHA-256 分别在 `configs/h5_protocol.lock.yaml` 和 `configs/h5_protocol.lock.sha256`。由于 v2 改变了优化器更新粒度，协议 ID 已显式升级，不能把 v1 未完成运行与 v2 结果混合报告。
