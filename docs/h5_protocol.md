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

## development gate

两个 backbone 都必须满足：primary 的 IDF1 与 HOTA 相对 baseline 不劣超过 0.002；IDSW 严格减少；至少 4 个 development 视频不受损。未通过则输出 `STOP_OR_REVISE_BEETRACKQUERY`，不解锁 final test。

完整机器可读协议与 SHA-256 分别在 `configs/h5_protocol.lock.yaml` 和 `configs/h5_protocol.lock.sha256`。
