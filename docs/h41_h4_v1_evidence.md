# H4-v1 实证复核与 H4.1 立项依据

## 不改写的原结论

真实 H4-v1 development 审计已经按冻结协议停止：

- ResNet50：151 个 baseline IDSW，精确 `t+10` joint recoverable 为 16/151（10.60%）；
- DINOv3：139 个 baseline IDSW，精确 `t+10` joint recoverable 为 8/139（5.76%）；
- 两者均低于 30% 门槛，因此 `STOP_NO_RECOVERABILITY_SIGNAL` 合法且继续有效；
- H4-v1 没有运行阶段 B，不存在可解释为方法结果的 assignment。

H4.1 不会修改 `configs/h4_protocol.lock.yaml`、H4-v1 输出或上述判断。运行前必须读取保存完好的 H4-v1 目录，并逐事件复核原有 1/3/5/10 帧结果。

## 为什么仍值得做 H4.1

H4-v1 计算的是精确时点统计。令事件总数为 (N)，事件 (i) 在精确延迟 (h) 是否可恢复为 (r_{i,h}\in\{0,1\})：

```text
R_exact(H) = (1/N) * Σ_i r_i,H
```

但“延迟决策能否利用短暂出现的证据”对应窗口统计：

```text
R_ever(≤H) = (1/N) * Σ_i max_{1≤h≤H} r_i,h
```

二者回答不同问题。一个事件可能在 `t+1` 可恢复，却因目标随后离开画面而在 `t+10` 不可用；这不应被解释为 `t+1` 的证据从未存在。

对用户提供的 H4-v1 事件表进行只读复核后，在**仅有的采样时点** `{1,3,5,10}` 上取并集，已经得到：

| Backbone（骨干网络） | 截至 H=1 | 截至 H=3 | 截至 H=5 | 截至 H=10 | 只在 H>1 首次恢复 | 只在 H=1 恢复 |
|---|---:|---:|---:|---:|---:|---:|
| ResNet50 | 46/151 = 30.46% | 48/151 = 31.79% | 53/151 = 35.10% | 58/151 = 38.41% | 12 | 20 |
| DINOv3 | 46/139 = 33.09% | 49/139 = 35.25% | 53/139 = 38.13% | 53/139 = 38.13% | 7 | 22 |

这只是连续 `t+1…t+10` 窗口统计的下界，却已显示两点：

1. 证据经常是短暂的，H=1 后可能消失；
2. 部分事件的证据异步出现，只检查 H=1 也会漏掉。

因此 H4.1 的合理问题不是“推翻 H4-v1”，而是“能否在最迟 H=5 的因果窗口中捕获短暂证据，并在证据稳定时提前提交”。

## 设计时序与统计身份

H4.1 是看到 H4-v1 development 结果后提出的，协议明确写为：

```text
FROZEN_EXPLORATORY_DEVELOPMENT
DEFINED_AFTER_OBSERVING_H4_V1_DEVELOPMENT_AUDIT
```

所以 H4.1 结果只能用于探索性机制判断和决定是否另建确认性协议。不得声称 H4.1 是 H4-v1 运行前预注册的，也不得在 H4.1 中读取 final test。

## 外部证据传输校验

本地分析使用的三份外部文件未提交仓库，其 SHA-256 为：

- `h4-v1-baseline-assignments.csv.gz`：`b790b715fce93f8ff707e74f9698e5e2ecdc0dc2f0a46a6739fa71dd277c703f`
- `h4-v1-recoverability-events.csv.gz`：`c827d822d2039243b833a38d53d7c4e74cb18c500dedca6d58c46d03e1fc60bb`
- `h4-v1-recoverability-audit.tar.gz`：`1c6c2fff07e60d232fc4ead3740ac4523a19a9990646f139b23a88924354268a`

服务器实际运行以未压缩源目录中的 metadata 自带 hash 为准；H4.1 不依赖这些下载路径。
