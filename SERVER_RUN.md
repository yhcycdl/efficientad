# 服务器多卡运行指南

推荐把项目放到多张 RTX 3090 的服务器上跑。3090 有 24GB 显存，比本地 8GB RTX 5060 宽裕很多；本项目不需要复杂的 DDP，多卡最稳做法是“一个训练进程占一张卡”，并行跑不同类别和模型。

## 1. 上传项目

在本地项目目录外执行：

```bash
scp -r EfficientAD user@server:/path/to/
```

或用 `rsync`：

```bash
rsync -av --exclude datasets --exclude results --exclude outputs EfficientAD/ user@server:/path/to/EfficientAD/
```

数据集、结果和输出目录可以不从本地传，服务器上重新下载/生成。

## 2. 检查服务器

登录服务器后：

```bash
cd /path/to/EfficientAD
nvidia-smi
```

确认能看到多张 3090。3090 常见环境建议用 CUDA 12.1 或 CUDA 11.8 的 PyTorch/Anomalib wheel；如果服务器驱动很新，也可以用 CUDA 13.0 extra。Anomalib 官方支持 `anomalib[cu130]`，同时文档也列出 `cu121` / `cu118` 后端选项；PyTorch 官方建议按官网 selector 选择适合 Linux + pip + CUDA 的安装命令。

## 3. 建环境

推荐先试 CUDA 12.1：

```bash
conda create -n efficientad-ad python=3.11 -y
conda activate efficientad-ad
pip install -U pip wheel setuptools
pip install "anomalib[cu121]" gradio scipy matplotlib
python scripts/check_env.py
```

如果服务器驱动或镜像不支持 `cu121`，改用：

```bash
pip install "anomalib[cu118]" gradio scipy matplotlib
```

如果服务器驱动很新且你希望用 CUDA 13.0：

```bash
pip install "anomalib[cu130]" gradio scipy matplotlib
```

也可以直接运行自动安装脚本：

```bash
bash scripts/server_setup.sh cu121
conda activate efficientad-ad
python scripts/check_env.py
```

如果服务器访问 HuggingFace 慢或报超时，先用镜像把 PatchCore 的预训练 backbone 缓存下来：

```bash
USE_HF_MIRROR=1 bash scripts/precache_models.sh --models patchcore
```

如果 Anomalib 自动下载 MVTec AD 报 `HTTP Error 404`，说明 Anomalib 内置的数据集下载链接不可用。先用官方分类别链接下载本项目需要的三类：

```bash
bash scripts/download_mvtec_categories.sh
```

下载完成后目录应类似：

```text
datasets/MVTecAD/bottle/train
datasets/MVTecAD/bottle/test
datasets/MVTecAD/hazelnut/train
datasets/MVTecAD/metal_nut/train
```

可以先检查 Anomalib 是否能读到非空 train/test split：

```bash
python scripts/check_datamodule.py --category all
```

如果服务器完全不能访问外网，不要在服务器上等下载。改成在有网机器上准备好下面两类资源，再传到服务器：

- `datasets/MVTecAD/`：MVTec AD 数据集，至少包含 `bottle`、`hazelnut`、`metal_nut`
- HuggingFace/timm cache：PatchCore 的 `wide_resnet50_2` 预训练权重缓存

有网机器上预缓存 PatchCore 权重：

```bash
conda activate efficientad-ad
python scripts/precache_models.py --models patchcore
tar -czf hf_cache.tar.gz -C ~/.cache huggingface
```

上传到服务器：

```bash
rsync -av datasets/MVTecAD/ user@server:/path/to/efficientad/datasets/MVTecAD/
scp hf_cache.tar.gz user@server:/path/to/efficientad/
```

服务器解压缓存：

```bash
cd /path/to/efficientad
mkdir -p ~/.cache
tar -xzf hf_cache.tar.gz -C ~/.cache
bash scripts/check_offline_assets.sh
```

离线跑完整实验：

```bash
OFFLINE=1 PRECACHE_MODELS=0 GPUS=0,1,2,3 EFFICIENTAD_PRESET=initial50 bash scripts/run_full_experiment_parallel.sh
```

## 4. 一次性跑完整实验

假设服务器有 4 张 GPU：

```bash
conda activate efficientad-ad
GPUS=0,1,2,3 EFFICIENTAD_PRESET=final100 bash scripts/run_full_experiment_parallel.sh
```

如果 HuggingFace 直连不稳定，运行完整实验时也加上镜像变量：

```bash
USE_HF_MIRROR=1 GPUS=0,1,2,3 EFFICIENTAD_PRESET=initial50 bash scripts/run_full_experiment_parallel.sh
```

如果今天时间紧，先跑 20 或 50 epoch：

```bash
GPUS=0,1,2,3 EFFICIENTAD_PRESET=initial50 bash scripts/run_full_experiment_parallel.sh
```

脚本会执行：

- `bottle/hazelnut/metal_nut + PatchCore`
- `bottle/hazelnut/metal_nut + EfficientAD`
- 三类 EfficientAD 的 `fixed / otsu / percentile` 阈值评估
- 每类保存若干 heatmap / overlay / mask 可视化

结果目录：

```text
results/
outputs/eval/
outputs/full_experiment_logs/
```

## 5. Demo

训练完成后找 checkpoint：

```bash
find results/efficientad -name "latest.ckpt" -o -name "*.ckpt" | sort
```

启动 demo：

```bash
python demo.py \
  --model efficientad \
  --ckpt results/efficientad/bottle/latest.ckpt \
  --server-name 0.0.0.0 \
  --server-port 7860
```

如果服务器不能直接打开端口，在本地做 SSH 转发：

```bash
ssh -L 7860:127.0.0.1:7860 user@server
```

然后本地浏览器打开 `http://127.0.0.1:7860`。

## 6. 运行策略

- 不建议一开始就 `final200`。今天要完整交付时，优先 `initial50` 或 `final100`，跑通后再补 `final200`。
- PatchCore 一般训练很快，EfficientAD 是主要耗时部分。
- 多张 3090 并行跑类别比 DDP 更稳，日志也更好整理。
- 如果服务器多人共用，先用 `nvidia-smi` 找空卡，再设置 `GPUS=空卡列表`。
