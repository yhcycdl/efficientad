# 报告写作提示

## 项目范围

本项目采用无监督工业视觉异常检测设定，训练阶段只使用正常样本，测试阶段同时判断图像级异常并定位像素级缺陷区域。实验只选择 MVTec AD 中的 `bottle`、`hazelnut`、`metal_nut` 三个类别，以保证在单人完成和本地 8GB 显存条件下能完成完整训练、评估和展示。

## 模型选择

PatchCore 作为 baseline，优点是小样本异常检测效果稳定、训练成本相对较低。EfficientAD 作为主模型，重点体现轻量、快速和较新的异常检测路线。两者均使用 Anomalib 框架实现，项目重点放在完整实验系统、指标对比和缺陷定位展示，而不是从零复现网络结构。

为了让对比更完整，可以将模型分为三层：PaDiM 和 STFPM 作为较弱/传统 baseline，PatchCore 作为强 baseline，EfficientAD 作为主模型。这样报告中既能证明 EfficientAD 相比传统方法有竞争力，也能诚实呈现 PatchCore 在 MVTec AD 上的强定位能力。

## 改进点表述

本文不声称提出新的深度模型，而是在模型输出的 anomaly map 上进行轻量级后处理优化与阈值策略消融。具体包括 Gaussian smoothing、固定阈值、Otsu 阈值、train-normal percentile 阈值、validation best-F1 阈值、多尺度 anomaly map 融合，以及连通域/形态学 mask 过滤。该设计更贴近工业部署中的后处理流程，也便于分析定位 mask 的稳定性。

增强版改进可以命名为“多尺度融合与连通域约束的缺陷定位后处理”。核心思路是在多个输入尺度下分别得到 anomaly map，将它们对齐后平均融合，再通过阈值分割、闭运算和小连通域过滤得到更稳定的 binary defect mask。该方法不改动 EfficientAD 主干网络，属于部署友好的后处理优化。

## 防止数据泄露

MVTec AD 的官方训练集只有正常图像，异常样本和像素级 mask 只出现在测试集中。因此，如果使用 best-F1 threshold，不能在全部 test 上寻找最优阈值后又在同一个 test 上汇报最终 F1。

推荐写法：

> 为避免阈值选择造成测试集信息泄露，本文将官方测试集按固定随机种子划分为 threshold validation 和 final test。threshold validation 仅用于选择 best-F1 threshold，最终指标均在未参与阈值搜索的 final test 上计算。

如果时间不足，可以不报告 best-F1 threshold，只报告 fixed、Otsu 和 train-normal percentile 三种不使用测试答案调阈值的策略。

## 未来工作

未来可以扩展到 VisA 数据集，增加更多类别；也可以加入 PaDiM、FastFlow 等方法进行更全面对比；进一步还可以尝试多尺度 anomaly map 融合或更复杂的形态学后处理。当前版本优先保证完整、可复现和可展示。
