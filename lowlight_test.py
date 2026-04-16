import argparse
import csv
import json
import os

import numpy as np
import torch
import torchvision
from PIL import Image

import model


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path)


def collect_images(image_dir):
    paths = []
    for name in sorted(os.listdir(image_dir)):
        if name.lower().endswith(IMAGE_EXTENSIONS):
            paths.append(os.path.join(image_dir, name))
    return paths


def load_tensor(image_path, device):
    image = Image.open(image_path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).float().to(device)
    return tensor


def save_json(path, payload):
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def run_test(config):
    device = torch.device("cuda" if torch.cuda.is_available() and not config.cpu else "cpu")
    DCE_net = model.enhance_net_nopool().to(device)
    DCE_net.load_state_dict(torch.load(config.checkpoint, map_location=device))
    DCE_net.eval()

    ensure_dir(config.output_dir)
    image_output_dir = os.path.join(config.output_dir, "images")
    ensure_dir(image_output_dir)

    rows = []
    with torch.no_grad():
        for image_path in collect_images(config.input_dir):
            data_lowlight = load_tensor(image_path, device)
            enhanced_image, _, meta = DCE_net(data_lowlight)
            out_name = os.path.splitext(os.path.basename(image_path))[0] + ".png"
            result_path = os.path.join(image_output_dir, out_name)
            torchvision.utils.save_image(enhanced_image, result_path)
            rows.append(
                {
                    "file_name": out_name,
                    "input_mean": meta["input_mean"],
                    "beta": meta["beta"],
                    "alpha": meta["alpha"],
                    "num_iter": meta["num_iter"],
                }
            )
            print("Saved:", result_path)

    with open(os.path.join(config.output_dir, "inference_meta.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    save_json(
        os.path.join(config.output_dir, "inference_summary.json"),
        {
            "checkpoint": os.path.abspath(config.checkpoint),
            "input_dir": os.path.abspath(config.input_dir),
            "num_images": len(rows),
            "avg_num_iter": sum(row["num_iter"] for row in rows) / max(1, len(rows)),
            "avg_input_mean": sum(row["input_mean"] for row in rows) / max(1, len(rows)),
            "avg_beta": sum(row["beta"] for row in rows) / max(1, len(rows)),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/home/danqihu/danqihu/504/Zero-DiDCE_reproduction/data/lol_dataset/lol_dataset/eval15/low",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )
    parser.add_argument("--cpu", action="store_true")
    run_test(parser.parse_args())

		
