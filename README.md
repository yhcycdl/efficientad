# EfficientAD 工业视觉异常检测与缺陷定位

这是课程大作业的可运行工程版本：使用 MVTec AD 的 `bottle`、`hazelnut`、`metal_nut` 三类，基于 Anomalib 跑通 PatchCore baseline 与 EfficientAD 主模型，并补充 Gaussian smoothing、阈值策略对比、可视化和 Gradio demo。

## 1. 环境

不要直接用当前系统 Python 3.13，建议新建 Python 3.11 环境：

```bash
conda env create -f environment.yml
conda activate efficientad-ad
python scripts/check_env.py
```

如果有多张 RTX 3090 的服务器，优先看 [SERVER_RUN.md](SERVER_RUN.md)，直接用服务器并行跑完整实验。

PatchCore 第一次运行会下载 `wide_resnet50_2` 预训练权重。如果服务器访问 HuggingFace 慢，先执行：

```bash
USE_HF_MIRROR=1 bash scripts/precache_models.sh --models patchcore
```

MVTec AD 数据集如果由 Anomalib 自动下载时报 404，先执行：

```bash
bash scripts/download_mvtec_categories.sh
```

如果 `anomalib[cu130]` 安装失败，可以先安装 PyTorch 官网推荐的 CUDA wheel，再执行：

```bash
pip install anomalib gradio matplotlib pandas pillow scikit-learn scipy
```

## 2. 第一天只跑通 smoke test

先不要跑 100/200 epoch。第一天目标是：

- `bottle + PatchCore` 跑通；
- `bottle + EfficientAD` 跑通 5 epoch；
- 能生成 anomaly heatmap。

推荐先执行：

```bash
bash scripts/day1_cli_smoke.sh
```

如果 CLI 参数因为 Anomalib 版本差异失败，再使用项目封装脚本：

```bash
bash scripts/day1_smoke.sh
```

也可以分开执行：

```bash
python train.py --model patchcore --category bottle --preset env
python train.py --model efficientad --category bottle --preset smoke
```

## 3. 训练节奏

脚本支持分档 preset：

```bash
# 环境验证：1 epoch
python train.py --model efficientad --category bottle --preset env

# smoke test：5 epoch
python train.py --model efficientad --category bottle --preset smoke

# 初版结果：20 或 50 epoch
python train.py --model efficientad --category bottle --preset initial20
python train.py --model efficientad --category bottle --preset initial50

# 最终实验：100 或 200 epoch
python train.py --model efficientad --category bottle --preset final100
python train.py --model efficientad --category bottle --preset final200
```

完整三类实验：

```bash
python train.py --model patchcore --category all --preset env
python train.py --model efficientad --category all --preset final100
```

服务器上如果 PatchCore 已经跑完，只重跑 EfficientAD 和评估：

```bash
USE_HF_MIRROR=1 PREPARE_MVTEC=0 RUN_PATCHCORE=0 GPUS=0,1,2,3 EFFICIENTAD_PRESET=initial20 bash scripts/run_full_experiment_parallel.sh
```

RTX 5060 8GB 如遇 OOM，优先把 batch size 降低：

```bash
python train.py --model efficientad --category bottle --preset smoke --train-batch-size 1 --eval-batch-size 1
python train.py --model patchcore --category bottle --preset env --train-batch-size 4 --eval-batch-size 4
```

## 4. 推理与可视化

训练完成后找到 checkpoint：

```bash
find results -name "*.ckpt" | sort
```

单张或文件夹推理：

```bash
python infer.py \
  --model efficientad \
  --ckpt results/efficientad/bottle/latest.ckpt \
  --input datasets/MVTecAD/bottle/test/broken_large/000.png \
  --output-dir outputs/infer/efficientad/bottle \
  --smooth-sigma 4 \
  --threshold-strategy otsu
```

输出包括：

- anomaly map `.npz`
- heatmap
- overlay
- binary mask
- side-by-side 对比图
- `predictions.csv`

## 5. 评估与阈值策略

基础阈值策略：

```bash
python eval.py \
  --model efficientad \
  --category bottle \
  --ckpt results/efficientad/bottle/latest.ckpt \
  --threshold-strategies fixed otsu percentile \
  --save-visuals
```

增强后处理优化实验：

```bash
python eval.py \
  --model efficientad \
  --category bottle \
  --output-dir outputs/eval_efficientad_fusion_morph \
  --fusion-scales 224 256 288 \
  --threshold-strategies fixed otsu percentile best_f1 \
  --threshold-val-ratio 0.2 \
  --mask-postprocess morph_cc \
  --min-area 64 \
  --close-size 5 \
  --save-visuals
```

该实验包含多尺度 anomaly map 融合、Gaussian smoothing、阈值策略和连通域/形态学 mask 过滤，可作为报告中的主要后处理改进。

额外弱 baseline 与轻量化/速度汇总：

```bash
bash scripts/run_extra_baselines.sh
python scripts/summarize_experiments.py
cat outputs/report_summary/compact_metrics.csv
cat outputs/report_summary/model_profiles.csv
```

默认会补跑 `PaDiM` 和 `STFPM`。PaDiM 是传统统计特征 baseline，STFPM 是 teacher-student baseline；PatchCore 保留为强 baseline，EfficientAD 是主模型。

如果需要补充“更老的深度模型”对照，建议先跑 `FastFlow` 和 `CFlow`：

```bash
USE_HF_MIRROR=1 GPUS=0,1,2 BASELINES=fastflow,cflow CATEGORIES=bottle,hazelnut,metal_nut bash scripts/run_extra_baselines.sh
python scripts/summarize_experiments.py
cat outputs/report_summary/final_comparison.csv
```

这两个模型用于报告中的历史 flow-based 深度 baseline 对比。`DRAEM` 也已支持，但它可能需要额外的 DTD anomaly source；如果想补重建式 baseline，可以单独跑：

```bash
USE_HF_MIRROR=1 GPUS=0,1,2 BASELINES=draem CATEGORIES=bottle,hazelnut,metal_nut bash scripts/run_extra_baselines.sh
```

若 `DRAEM` 在当前 Anomalib 版本中因为数据源或接口差异报错，可以直接换成备用的 `ReverseDistillation`：

```bash
USE_HF_MIRROR=1 GPUS=0,1,2 BASELINES=reverse_distillation CATEGORIES=bottle,hazelnut,metal_nut bash scripts/run_extra_baselines.sh
```

如果要做 validation best-F1 threshold，必须避免测试集信息泄露：

```bash
python eval.py \
  --model efficientad \
  --category bottle \
  --ckpt results/efficientad/bottle/latest.ckpt \
  --threshold-strategies fixed otsu percentile best_f1 \
  --threshold-val-ratio 0.2 \
  --save-visuals
```

报告中建议写明：

> 为避免阈值选择造成测试集信息泄露，本文将官方测试集按固定随机种子划分为 threshold validation 和 final test。验证集仅用于选择 best-F1 threshold，最终指标均在未参与阈值搜索的 final test 上计算。

时间不够时，不做 `best_f1`，只保留 `fixed`、`otsu`、`percentile` 更稳。

## 6. Gradio demo

```bash
python demo.py \
  --model efficientad \
  --ckpt results/efficientad/bottle/latest.ckpt \
  --threshold-strategy otsu \
  --smooth-sigma 4
```

打开命令行显示的本地 URL，上传图片后会展示异常分数、heatmap、overlay 和 mask。

## 7. 项目结构

```text
.
├── train.py              # Anomalib 训练封装
├── infer.py              # 单图/文件夹推理与可视化
├── eval.py               # 指标汇总与阈值策略对比
├── postprocess.py        # Gaussian smoothing、Otsu、percentile、best-F1
├── visualize.py          # heatmap/overlay/mask 生成
├── demo.py               # Gradio demo
├── common.py             # 模型、数据、路径等公共工具
├── data_config.py        # 类别、路径、实验 preset
├── scripts/
│   ├── check_env.py
│   └── day1_smoke.sh
├── datasets/             # MVTec AD 下载目录
├── results/              # 训练 checkpoint 和日志
└── outputs/              # 推理、评估、可视化结果
```

## 8. 报告定位

改进点建议表述为“轻量级后处理优化与阈值策略消融”，不要写成全新算法。主线报告结构：

1. 工业异常检测背景与无监督设定；
2. PatchCore 与 EfficientAD 方法简介；
3. MVTec AD 三类实验设置；
4. Gaussian smoothing 与阈值策略；
5. image AUROC、pixel AUROC、pixel F1、推理时间对比；
6. heatmap、overlay、mask 和失败案例分析；
7. demo 展示与未来工作。
