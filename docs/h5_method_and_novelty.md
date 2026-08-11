# BeeTrackQuery 方法来源与创新边界

BeeTrackQuery 借鉴 DETR 系 tracking（基于查询的跟踪）和 DETRAM 的 persistent tracking query（持久跟踪查询）、current-frame query refresh（当前帧查询刷新）与短期 memory（记忆）。这些结构本身属于已有思想，不能作为本项目的独占创新点。DETRAM 报告的短记忆优于过长陈旧记忆，也直接影响本实验把 memory length（记忆长度）冻结为 4，而不是无限累积。

本项目研究贡献应被表述为 BEE24 场景中的机制验证与问题建模：极端 appearance homogeneity（外观同质性）下，单纯替换 backbone 不足；因此把 identity state（身份状态）从单帧 crop 表征中分离出来，并让“是否写入身份状态”成为有监督、可校准、可审计的决策。可信度门只控制记忆写入，不读取 GT、不延迟到未来帧、不运行多假设 oracle（预知真值诊断）。

与 TOPIC 的边界也必须清楚：TOPIC 的官方 Re-ID checkpoint 与 BEE24 训练数据有重叠，只能作为参考证据；它不是 BeeTrackQuery 组件。BeeTrackQuery 不是在 TOPIC 上附加小模块，也不复用其 detector/tracker。

目前代码阶段的真实 GPU、完整 BEE24 与固定 detector 验证均为 `SERVER_VALIDATION_PENDING`。GT boxes（真值框）结果只说明关联机制上限，不等于端到端 MOT。
