# H3 frozen protocol

H3 的机器可读冻结文件是 [`configs/h3_protocol.lock.yaml`](../configs/h3_protocol.lock.yaml)，对应 SHA-256 在 [`configs/h3_protocol.lock.sha256`](../configs/h3_protocol.lock.sha256)。任何内容变化而未经过明确重新冻结都会被 `h3-validate-protocol` 拒绝。

## 已冻结内容

- video-level `project_train / development_validation / final_test` 隔离；
- identity 固定为 `(video_id, track_id)`；
- final test 不参与训练、阈值、模型选择和 ablation；
- reliability 只使用身份历史离群、尺度变化、方向代理变化、清晰度变化和拥挤/重叠；
- 静态 bbox area 与静态 Laplacian 排除；
- 所有 deployable threshold 在 `project_train` 拟合，development 只做选择与消融；
- 先 GT detection boxes，再 fixed detector boxes；
- `baseline association / selective memory update / reliability-weighted association / full RAM-Bee` 四组；
- primary metrics 为 AssA、IDF1、IDSW、Frag、HOTA；
- official TOPIC AGW 继续是 reference-only，split-clean AGW 尚待实现与服务器验证。

## final-test 门禁

只有 method code、config、checkpoint、threshold artifact 和 reporting code 全部冻结，且 development 结果已签字确认后，才允许一次 final-test 运行。任何提前读取、未来信息泄漏、split/identity 泄漏或把官方 AGW 声称为 split-clean，都触发 STOP。

当前实现已交付 H3 协议验证器、冻结门禁和第一阶段 GT-box online tracker。GT-box 阶段包含 project-train 阈值拟合、四种关联/记忆变体和 identity metrics，但 fixed-detector adapter、真实服务器结果与 final test 仍未完成。H2.5 paired benefit 只是进入 H3 的 GO 证据之一；真正的论文主张仍必须由 GT-box 与 fixed-detector development 共同支持。

验证命令：

```bash
~/venvs/beeid-dino/bin/python -m beeid.cli h3-validate-protocol \
  --protocol configs/h3_protocol.lock.yaml \
  --checksum configs/h3_protocol.lock.sha256
```

实现和服务器运行分别见 [h3_implementation.md](h3_implementation.md) 与 [h3_server_deployment.md](h3_server_deployment.md)。
