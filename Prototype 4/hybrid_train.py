import argparse
import os
import time

import torch
import torch.nn as nn

import hybrid_dataloader
import hybrid_loss
import hybrid_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def weights_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def format_loss_dict(loss_dict):
    parts = []
    for key in sorted(loss_dict.keys()):
        parts.append("{}={:.4f}".format(key, loss_dict[key].item()))
    return ", ".join(parts)


def save_checkpoint(model, optimizer, epoch, step, path, config):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "config": vars(config),
    }
    torch.save(checkpoint, path)


def maybe_resume(model, optimizer, resume_path):
    checkpoint = torch.load(resume_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = checkpoint.get("epoch", 0)
    start_step = checkpoint.get("step", 0)
    return start_epoch, start_step


def train(config):
    model = hybrid_model.HybridZeroDCE(
        base_channels=config.base_channels,
        curve_steps=config.curve_steps,
    ).to(DEVICE)
    model.apply(weights_init)

    if config.load_zero_dce_pretrain and os.path.exists(config.zero_dce_pretrain):
        matched, skipped = model.load_zero_dce_pretrain(config.zero_dce_pretrain, map_location=DEVICE)
        print("Loaded {} Zero-DCE parameters from {}".format(len(matched), config.zero_dce_pretrain))
        if config.verbose and skipped:
            print("Skipped {} Zero-DCE parameters".format(len(skipped)))

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = hybrid_loss.HybridZeroDCELoss(
        exp_patch_size=config.exp_patch_size,
        exp_mean_val=config.exp_mean_val,
        weight_color=config.weight_color,
        weight_spa=config.weight_spa,
        weight_exp=config.weight_exp,
        weight_curve_tv=config.weight_curve_tv,
        weight_reconstruction=config.weight_reconstruction,
        weight_base_identity=config.weight_base_identity,
        weight_illumination=config.weight_illumination,
        weight_reflectance_grad=config.weight_reflectance_grad,
        weight_residual=config.weight_residual,
    ).to(DEVICE)

    start_epoch = 0
    global_step = 0
    if config.resume and os.path.exists(config.resume):
        start_epoch, global_step = maybe_resume(model, optimizer, config.resume)
        print("Resumed from {} at epoch {}, step {}".format(config.resume, start_epoch, global_step))

    train_dataset = hybrid_dataloader.LowLightDataset(
        config.lowlight_images_path,
        image_size=config.image_size,
        random_flip=not config.disable_flip,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=(DEVICE.type == "cuda"),
    )

    os.makedirs(config.snapshots_folder, exist_ok=True)
    model.train()

    for epoch in range(start_epoch, config.num_epochs):
        epoch_start = time.time()
        for iteration, img_lowlight in enumerate(train_loader, start=1):
            global_step += 1
            img_lowlight = img_lowlight.to(DEVICE, non_blocking=(DEVICE.type == "cuda"))

            outputs = model(img_lowlight)
            total_loss, loss_dict = criterion(outputs, img_lowlight)

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()

            if global_step % config.display_iter == 0:
                print(
                    "Epoch {}/{} Step {} Total {:.4f} | {}".format(
                        epoch + 1,
                        config.num_epochs,
                        global_step,
                        total_loss.item(),
                        format_loss_dict(loss_dict),
                    )
                )

            if global_step % config.snapshot_iter == 0:
                snapshot_path = os.path.join(config.snapshots_folder, "iter_{:07d}.pth".format(global_step))
                save_checkpoint(model, optimizer, epoch + 1, global_step, snapshot_path, config)

        latest_path = os.path.join(config.snapshots_folder, "latest.pth")
        save_checkpoint(model, optimizer, epoch + 1, global_step, latest_path, config)
        print("Finished epoch {} in {:.2f}s".format(epoch + 1, time.time() - epoch_start))


def parse_args():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    eecs504_root = os.path.dirname(project_root)
    zero_dce_root = os.path.abspath(os.path.join(current_dir, "..", "..", "Zero-DCE", "Zero-DCE_code"))
    default_train_root = os.path.join(eecs504_root, "train_data")

    parser = argparse.ArgumentParser(description="Train a KinD-inspired Zero-DCE hybrid model with zero-reference losses.")
    parser.add_argument("--lowlight_images_path", type=str, default=default_train_root)
    parser.add_argument("--snapshots_folder", type=str, default=os.path.join(current_dir, "snapshots_hybrid"))
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--zero_dce_pretrain", type=str, default=os.path.join(zero_dce_root, "snapshots", "Epoch99.pth"))
    parser.add_argument("--load_zero_dce_pretrain", dest="load_zero_dce_pretrain", action="store_true")
    parser.add_argument("--disable_zero_dce_pretrain", dest="load_zero_dce_pretrain", action="store_false")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip_norm", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--display_iter", type=int, default=10)
    parser.add_argument("--snapshot_iter", type=int, default=200)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--disable_flip", action="store_true")

    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--curve_steps", type=int, default=8)
    parser.add_argument("--exp_patch_size", type=int, default=16)
    parser.add_argument("--exp_mean_val", type=float, default=0.6)

    parser.add_argument("--weight_color", type=float, default=5.0)
    parser.add_argument("--weight_spa", type=float, default=1.0)
    parser.add_argument("--weight_exp", type=float, default=10.0)
    parser.add_argument("--weight_curve_tv", type=float, default=200.0)
    parser.add_argument("--weight_reconstruction", type=float, default=3.0)
    parser.add_argument("--weight_base_identity", type=float, default=1.0)
    parser.add_argument("--weight_illumination", type=float, default=0.25)
    parser.add_argument("--weight_reflectance_grad", type=float, default=0.2)
    parser.add_argument("--weight_residual", type=float, default=0.05)
    parser.set_defaults(load_zero_dce_pretrain=True)
    return parser.parse_args()


if __name__ == "__main__":
    config = parse_args()
    print("Using device:", DEVICE)
    train(config)
