# H4 方法边界与相关工作定位

## 我们不讲的故事

H4 不把“在 TOPIC 上加一个小模块”作为创新，也不以“DINOv3 比 CNN 强”为目标。TOPIC 继续只是 BEE24 上的外部系统参考；H4 源码不 import、修改或调用 TOPIC tracker。ResNet50 与 DINOv3 都是冻结证据源，作用是检查机制是否跨 backbone 成立。

## 已有工作覆盖了什么

- Multiple Hypothesis Tracking 很早就是多目标跟踪中的经典思想；例如 Cox 与 Hingorani 的 MHT 系列以及后续的视觉 MHT。不能把“保留多个假设”本身写成贡献。
- [Tracking by Associating Clips, ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/5114_ECCV_2022_paper.php) 已明确用多帧 clip association 避免逐帧错误传播。
- [MeMOT, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Cai_MeMOT_Multi-Object_Tracking_With_Memory_CVPR_2022_paper.html) 与 [MeMOTR, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Gao_MeMOTR_Long-Term_Memory-Augmented_Transformer_for_Multi-Object_Tracking_ICCV_2023_paper.html) 已研究长期记忆和时序聚合。
- [Robust Multi-Object Tracking by Marginal Inference, ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/317_ECCV_2022_paper.php) 已研究相对/概率化分配不确定性。
- [U2MOT, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Liu_Uncertainty-Aware_Unsupervised_Multi-Object_Tracking_ICCV_2023_paper.html) 已将 uncertainty 引入 MOT 学习与关联。
- [TOPIC, arXiv 2023](https://arxiv.org/abs/2308.11157) 的完整系统包含 detector、OC-SORT、AGW、并行匹配及 AARM；不能把官方 AGW 的高分简单归因于一个 appearance backbone。

## H4 真正检验的组合命题

潜在贡献不是某个单独组件，而是由 H1–H3 机制证据导出的受约束组合：

1. 先量化“当前不可判、未来可判”的事件比例；若不存在，不实现方法故事。
2. 只在局部 assignment 歧义时延迟，而非全视频固定 clip 化。
3. 未决身份假设携带各自独立的 appearance memory，错误分支不能污染获胜分支。
4. 用 frozen backbone 双重验证，避免把收益归因于重新训练 ReID。
5. 将 latency、假设数、oversized conflicts 与 accuracy 一起报告，明确方法代价。

如果实验成立，论文中的稳妥表述应是：

> 在高度同质、外观瞬时不可靠的蜂群跟踪中，身份错误的一部分具有短期未来可恢复性；对这些局部冲突延迟提交，并在未决期间隔离身份记忆，可以减弱立即提交造成的错误传播。

如果 oracle gate 或双 backbone 方法 gate 失败，应按冻结协议停止，而不是继续调阈值直到得到正结果。

## 与 TOPIC 的关系

TOPIC 可在后续固定 detector 阶段作为外部参考，用来回答“我们的 mechanism 是否在一个强 BEE24 系统之外仍有解释力”。但以下内容均不属于 H4：复刻 TOPIC detector/tracker、把 TOPIC checkpoint 用于 split-clean 方法选择、将 H4 命名为 TOPIC 改进版、或用普通 ResNet50 结果冒充 AGW。
