# H4.1 方法来源、贡献边界与论文叙事

## 思路来自哪里

H4.1 不是凭空发明“多帧跟踪”。它明确继承和约束以下已有思想：

- [Tracking by Associating Clips（ECCV 2022）](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/5114_ECCV_2022_paper.php)：多帧 clip association（片段关联）用于减少逐帧错误传播；
- [Robust Multi-Object Tracking by Marginal Inference（ECCV 2022）](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/317_ECCV_2022_paper.php)：显式处理 assignment uncertainty（分配不确定性）；
- [MeMOT（CVPR 2022）](https://openaccess.thecvf.com/content/CVPR2022/html/Cai_MeMOT_Multi-Object_Tracking_With_Memory_CVPR_2022_paper.html) 与 [MeMOTR（ICCV 2023）](https://openaccess.thecvf.com/content/ICCV2023/html/Gao_MeMOTR_Long-Term_Memory-Augmented_Transformer_for_Multi-Object_Tracking_ICCV_2023_paper.html)：时序记忆在 MOT 中的作用；
- 经典 MHT（Multiple Hypothesis Tracking，多假设跟踪）：在歧义解除前保留候选世界状态；
- 本项目 H2/H2.5：错误 observation 更新单一 EMA memory（指数滑动平均记忆）可能产生污染；
- 本项目真实 H4-v1：证据既会短暂消失，也会延后出现，精确 `t+10` 不是“窗口内曾可恢复”的有效替代。

因此 H4.1 的 adaptive rule（自适应规则）是对项目实证约束的工程化回答：只在局部歧义触发，证据稳定即提交，最迟 H=5，且未提交分支携带各自隔离记忆。

## 不能声称什么

以下都不是本项目创新：

- MHT、beam search（束搜索）、fixed-lag smoothing（固定延迟平滑）或 early stopping（提前停止）本身；
- 使用 appearance+motion（外观与运动）联合分数；
- 在 TOPIC 上添加一个小模块；H4.1 不 import、不修改也不运行 TOPIC tracker；
- DINOv3 比 ResNet50 更强；两个 frozen backbone 是跨表示机制验证，不是冠军竞赛；
- 用 GT oracle 取得可部署性能；oracle 只提供诊断上限。

## 只有结果支持后才能讨论的贡献

如果两个 backbone 都通过累计恢复门禁，且 adaptive-isolated 主变体同时减少 IDSW、保持 IDF1/HOTA 非劣，稳妥的贡献表述是：

> 在高度同质、单帧外观不稳定的蜜蜂跟踪中，身份判定证据呈短暂且异步出现；项目以事件触发、证据稳定即提交和候选记忆隔离的受约束组合，减少立即提交造成的身份错误传播。

这是一项由 H1–H4.1 因果诊断链支持的、面向同质动物跟踪的组合方法贡献，不是对已有多假设思想的所有权声明。

如果只通过 recoverability gate（可恢复门禁）但方法 gate 失败，故事应停止在“存在可利用窗口证据，但当前打分/搜索/提交规则未能实现收益”。如果累计门禁也失败，则应放弃这条主线，而不是继续事后调阈值。

## 为什么 H4.1 仍只是探索性

窗口 estimand、H=5 门禁和自适应提交规则是在看到 H4-v1 development 结果后冻结的。透明披露这一点比假装预注册更重要。H4.1 若成功，下一步是冻结一个新 confirmatory protocol（确认性协议），优先在固定 detector development 上验证；final test 仍只能在所有规则停止修改后按新协议运行一次。
