import argparse
import os
from glob import glob

import numpy as np
import torch
from PIL import Image

import hybrid_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE = Image.LANCZOS


def list_images(input_dir):
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    image_paths = []
    for pattern in patterns:
        image_paths.extend(glob(os.path.join(input_dir, pattern)))
    return sorted(image_paths)


def load_image(path, max_dim=0):
    image = Image.open(path).convert("RGB")
    if max_dim and max(image.size) > max_dim:
        image.thumbnail((max_dim, max_dim), RESAMPLE)
    image = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    return tensor, image.shape[0], image.shape[1]


def save_image(path, tensor):
    image = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    image = np.clip(image, 0.0, 1.0)
    image = (image * 255.0).round().astype(np.uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(image).save(path)


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint, strict=False)


def parse_args():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Run inference with the KinD-inspired Zero-DCE hybrid model.")
    parser.add_argument("--weights", required=True, help="Path to a trained hybrid model checkpoint.")
    parser.add_argument("--input_dir", default=os.path.join(current_dir, "test_data/low"))
    parser.add_argument("--output_dir", default=os.path.join(current_dir, "test_data/result"))
    parser.add_argument("--max_dim", type=int, default=0)
    parser.add_argument("--save_intermediate", action="store_true")
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--curve_steps", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    image_paths = list_images(args.input_dir)
    if not image_paths:
        raise RuntimeError("No input images found in {}".format(args.input_dir))

    model = hybrid_model.HybridZeroDCE(
        base_channels=args.base_channels,
        curve_steps=args.curve_steps,
    ).to(DEVICE)
    load_checkpoint(model, args.weights)
    model.eval()

    print("Using device:", DEVICE)
    print("Loaded weights:", args.weights)

    with torch.no_grad():
        for image_path in image_paths:
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            image_tensor, _, _ = load_image(image_path, max_dim=args.max_dim)
            image_tensor = image_tensor.to(DEVICE)
            outputs = model(image_tensor)

            save_image(os.path.join(args.output_dir, "{}_hybrid.png".format(image_name)), outputs["enhanced_image"])
            if args.save_intermediate:
                save_image(os.path.join(args.output_dir, "{}_base.png".format(image_name)), outputs["base_image"])
                save_image(os.path.join(args.output_dir, "{}_decomposed.png".format(image_name)), outputs["decomposed_image"])
                save_image(os.path.join(args.output_dir, "{}_reflectance.png".format(image_name)), outputs["reflectance_clean"])
                illumination = outputs["illumination"].repeat(1, 3, 1, 1)
                save_image(os.path.join(args.output_dir, "{}_illumination.png".format(image_name)), illumination)


if __name__ == "__main__":
    main()
