"""Gradio demo for uploading an image and visualizing anomaly localization."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from common import MODEL_CHOICES, build_engine, build_model, build_predict_dataset, resolve_checkpoint, scalar_value, tensor_to_numpy
from data_config import DEFAULT_FIXED_THRESHOLD, DEFAULT_IMAGE_SIZE, DEFAULT_SMOOTH_SIGMA
from postprocess import apply_threshold_strategy, prepare_map
from visualize import colorize_heatmap, make_overlay, mask_image, resize_map_to_image


def build_demo(args: argparse.Namespace):
    import gradio as gr

    checkpoint = resolve_checkpoint(args.ckpt)

    def predict(image: Image.Image):
        if image is None:
            return "请先上传图片。", None, None, None

        image = image.convert("RGB")
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "input.png"
            image.save(image_path)
            dataset = build_predict_dataset(image_path, image_size=args.image_size)
            model = build_model(args.model)
            engine = build_engine()
            predictions = engine.predict(model=model, dataset=dataset, ckpt_path=str(checkpoint))
            if not predictions:
                return "模型没有返回预测结果。", None, None, None
            pred = predictions[0]

        anomaly_map = tensor_to_numpy(getattr(pred, "anomaly_map", None))
        if anomaly_map is None:
            anomaly_map = tensor_to_numpy(getattr(pred, "pred_mask", None))
        if anomaly_map is None:
            return "预测结果中没有 anomaly map。", None, None, None

        score = scalar_value(getattr(pred, "pred_score", None))
        if score is None:
            score = float(np.max(anomaly_map))

        processed = prepare_map(anomaly_map, smooth_sigma=args.smooth_sigma, normalize=True)
        resized_map = resize_map_to_image(processed, image)
        mask, threshold = apply_threshold_strategy(
            resized_map,
            strategy=args.threshold_strategy,
            fixed_value=args.fixed_threshold,
        )
        heatmap = colorize_heatmap(resized_map)
        overlay = make_overlay(image, heatmap)
        mask_pil = mask_image(mask)
        label = "异常" if int(mask.max() > 0) else "正常"
        text = f"判断：{label}\n异常分数：{score:.6f}\n阈值策略：{args.threshold_strategy}\n阈值：{threshold:.6f}"
        return text, heatmap, overlay, mask_pil

    with gr.Blocks(title="工业异常检测 demo") as demo:
        gr.Markdown("# 工业异常检测与缺陷定位 Demo")
        with gr.Row():
            input_image = gr.Image(type="pil", label="上传产品图片")
            result_text = gr.Textbox(label="检测结果", lines=4)
        with gr.Row():
            heatmap = gr.Image(type="pil", label="Heatmap")
            overlay = gr.Image(type="pil", label="Overlay")
            mask = gr.Image(type="pil", label="Binary Mask")
        submit = gr.Button("检测")
        submit.click(predict, inputs=input_image, outputs=[result_text, heatmap, overlay, mask])
    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Gradio anomaly detection demo.")
    parser.add_argument(
        "--model",
        required=True,
        choices=MODEL_CHOICES,
    )
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--smooth-sigma", type=float, default=DEFAULT_SMOOTH_SIGMA)
    parser.add_argument("--threshold-strategy", choices=["fixed", "otsu"], default="otsu")
    parser.add_argument("--fixed-threshold", type=float, default=DEFAULT_FIXED_THRESHOLD)
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    demo = build_demo(args)
    demo.launch(server_name=args.server_name, server_port=args.server_port)


if __name__ == "__main__":
    main()
