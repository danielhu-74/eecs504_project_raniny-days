#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys

import numpy as np
from PIL import Image
from skimage import color
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

if not hasattr(np, "int"):
    np.int = int

PROJECT_ROOT = "/home/danqihu/danqihu/504"
PYDEPS_DIR = os.path.join(PROJECT_ROOT, ".pydeps")
if os.path.isdir(PYDEPS_DIR) and PYDEPS_DIR not in sys.path:
    sys.path.append(PYDEPS_DIR)

import scipy
from skvideo.measure import niqe


def _compat_imresize(array, size, interp="bicubic", mode=None):
    if isinstance(size, float):
        target_height = max(1, int(round(array.shape[0] * size)))
        target_width = max(1, int(round(array.shape[1] * size)))
    elif isinstance(size, (tuple, list)) and len(size) == 2:
        target_height = int(size[0])
        target_width = int(size[1])
    else:
        raise ValueError("Unsupported resize target: {}".format(size))

    resample_map = {
        "nearest": Image.NEAREST,
        "bilinear": Image.BILINEAR,
        "bicubic": Image.BICUBIC,
        "lanczos": Image.LANCZOS,
    }
    resample = resample_map.get(interp, Image.BICUBIC)

    if mode == "F" or array.dtype.kind == "f":
        pil_image = Image.fromarray(array.astype(np.float32), mode="F")
    else:
        pil_image = Image.fromarray(array)

    resized = pil_image.resize((target_width, target_height), resample=resample)
    return np.asarray(resized)


if hasattr(scipy, "misc") and not hasattr(scipy.misc, "imresize"):
    scipy.misc.imresize = _compat_imresize


def ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path)


def save_json(path, payload):
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def load_rgb(image_path):
    return np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0


def resolve_by_stem(folder, stem):
    for extension in (".png", ".jpg", ".jpeg", ".bmp"):
        candidate = os.path.join(folder, stem + extension)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError("Missing file for stem {} under {}".format(stem, folder))


def compute_psnr(reference_rgb, candidate_rgb):
    score = peak_signal_noise_ratio(reference_rgb, candidate_rgb, data_range=1.0)
    return float(score) if np.isfinite(score) else float("inf")


def compute_ssim(reference_rgb, candidate_rgb):
    return float(
        structural_similarity(
            reference_rgb,
            candidate_rgb,
            channel_axis=-1,
            data_range=1.0,
        )
    )


def compute_deltae(reference_rgb, candidate_rgb):
    reference_lab = color.rgb2lab(np.clip(reference_rgb, 0.0, 1.0))
    candidate_lab = color.rgb2lab(np.clip(candidate_rgb, 0.0, 1.0))
    delta = color.deltaE_ciede2000(reference_lab, candidate_lab)
    return float(np.mean(delta))


def downsample_lightness_map(image_rgb, max_side):
    lightness = np.max(np.clip(image_rgb, 0.0, 1.0), axis=2).astype(np.float32)
    height, width = lightness.shape
    scale = min(1.0, float(max_side) / float(max(height, width)))
    new_height = max(1, int(round(height * scale)))
    new_width = max(1, int(round(width * scale)))
    pil = Image.fromarray(lightness, mode="F")
    resized = pil.resize((new_width, new_height), Image.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def compute_loe(source_rgb, target_rgb, max_side=50):
    source_lightness = downsample_lightness_map(source_rgb, max_side)
    target_lightness = downsample_lightness_map(target_rgb, max_side)
    source_flat = source_lightness.reshape(-1)
    target_flat = target_lightness.reshape(-1)
    source_order = source_flat[:, None] >= source_flat[None, :]
    target_order = target_flat[:, None] >= target_flat[None, :]
    disagreement = np.not_equal(source_order, target_order)
    return float(np.mean(np.sum(disagreement, axis=1)))


def compute_niqe(candidate_rgb):
    gray = np.dot(candidate_rgb[..., :3], np.array([0.299, 0.587, 0.114], dtype=np.float32))
    gray_uint8 = np.clip(np.round(gray * 255.0), 0, 255).astype(np.uint8)
    score = niqe(gray_uint8)
    return float(np.asarray(score).reshape(-1)[0])


def aggregate(values):
    finite_values = [float(value) for value in values if np.isfinite(value)]
    return {
        "mean": float(np.mean(finite_values)),
        "std": float(np.std(finite_values)),
        "min": float(np.min(finite_values)),
        "max": float(np.max(finite_values)),
    }


def evaluate(config):
    ensure_dir(config.output_dir)
    prediction_dir = config.prediction_dir
    rows = []

    for image_name in sorted(os.listdir(prediction_dir)):
        if not image_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            continue
        stem = os.path.splitext(image_name)[0]
        pred_path = os.path.join(prediction_dir, image_name)
        gt_path = resolve_by_stem(config.gt_dir, stem)
        input_path = resolve_by_stem(config.input_dir, stem)

        pred_rgb = load_rgb(pred_path)
        gt_rgb = load_rgb(gt_path)
        input_rgb = load_rgb(input_path)

        rows.append(
            {
                "file_name": image_name,
                "psnr": compute_psnr(gt_rgb, pred_rgb),
                "ssim": compute_ssim(gt_rgb, pred_rgb),
                "deltae": compute_deltae(gt_rgb, pred_rgb),
                "loe": compute_loe(input_rgb, pred_rgb, max_side=config.loe_max_side),
                "niqe": compute_niqe(pred_rgb),
            }
        )

    if not rows:
        raise ValueError("No prediction images found in {}".format(prediction_dir))

    summary = {
        "num_images": len(rows),
        "psnr": aggregate([row["psnr"] for row in rows]),
        "ssim": aggregate([row["ssim"] for row in rows]),
        "deltae": aggregate([row["deltae"] for row in rows]),
        "loe": aggregate([row["loe"] for row in rows]),
        "niqe": aggregate([row["niqe"] for row in rows]),
    }

    csv_path = os.path.join(config.output_dir, "metrics_per_image.csv")
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    save_json(os.path.join(config.output_dir, "metrics_summary.json"), summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_dir", type=str, required=True)
    parser.add_argument(
        "--gt_dir",
        type=str,
        default="/home/danqihu/danqihu/504/Zero-DiDCE_reproduction/data/lol_dataset/lol_dataset/eval15/high",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/home/danqihu/danqihu/504/Zero-DiDCE_reproduction/data/lol_dataset/lol_dataset/eval15/low",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--loe_max_side", type=int, default=50)
    evaluate(parser.parse_args())
