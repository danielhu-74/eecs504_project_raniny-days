import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch

import Myloss
import dataloader
import model


def weights_init(module):
    classname = module.__class__.__name__
    if classname.find("Conv") != -1:
        module.weight.data.normal_(0.0, 0.02)
        if module.bias is not None:
            module.bias.data.zero_()


def ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path)


def save_json(path, payload):
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(config):
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not config.cpu else "cpu")
    print("Using device:", device)

    ensure_dir(config.snapshots_folder)
    ensure_dir(config.run_root)
    save_json(os.path.join(config.run_root, "train_config.json"), vars(config))

    DCE_net = model.enhance_net_nopool().to(device)
    DCE_net.apply(weights_init)
    if config.load_pretrain:
        DCE_net.load_state_dict(torch.load(config.pretrain_dir, map_location=device))

    train_dataset = dataloader.lowlight_loader(
        config.lowlight_images_path,
        image_size=config.image_size,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )

    piecewise_loss = Myloss.PiecewiseNonReferenceLoss(
        patch_size=config.patch_size,
        target_value=config.target_value,
        q1=config.q1,
        q2=config.q2,
        m=config.m_value,
        w1=config.w1,
        w2=config.w2,
    )
    tv_loss = Myloss.L_TV(tv_weight=1.0)
    optimizer = torch.optim.Adam(DCE_net.parameters(), lr=config.lr)

    DCE_net.train()
    rows = []
    best_loss = None

    for epoch in range(config.num_epochs):
        epoch_start = time.time()
        sums = {
            "loss_total": 0.0,
            "piecewise_total": 0.0,
            "l1_extreme": 0.0,
            "l2_general": 0.0,
            "tv_loss": 0.0,
            "extreme_ratio": 0.0,
            "enhanced_region_mean": 0.0,
            "input_region_mean": 0.0,
            "avg_num_iter": 0.0,
            "avg_input_mean": 0.0,
            "avg_beta": 0.0,
        }

        for iteration, img_lowlight in enumerate(train_loader):
            img_lowlight = img_lowlight.to(device)

            enhanced_image, curve_map, meta = DCE_net(img_lowlight)
            piecewise_total, stats = piecewise_loss(img_lowlight, enhanced_image)

            tv_total = torch.tensor(0.0, device=device)
            if config.tv_weight > 0.0:
                tv_total = config.tv_weight * tv_loss(curve_map)

            loss = piecewise_total + tv_total

            optimizer.zero_grad()
            loss.backward()
            if config.grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(DCE_net.parameters(), config.grad_clip_norm)
            optimizer.step()

            sums["loss_total"] += float(loss.detach().cpu())
            sums["piecewise_total"] += stats["piecewise_total"]
            sums["l1_extreme"] += stats["l1_extreme"]
            sums["l2_general"] += stats["l2_general"]
            sums["tv_loss"] += float(tv_total.detach().cpu())
            sums["extreme_ratio"] += stats["extreme_ratio"]
            sums["enhanced_region_mean"] += stats["enhanced_region_mean"]
            sums["input_region_mean"] += stats["input_region_mean"]
            sums["avg_num_iter"] += float(meta["num_iter"])
            sums["avg_input_mean"] += float(meta["input_mean"])
            sums["avg_beta"] += float(meta["beta"])

            if (iteration + 1) % config.display_iter == 0:
                print(
                    "Epoch {} Iter {} Loss {:.6f}".format(
                        epoch + 1,
                        iteration + 1,
                        float(loss.detach().cpu()),
                    )
                )

        num_steps = max(1, len(train_loader))
        row = {
            "epoch": epoch + 1,
            "loss_total": sums["loss_total"] / num_steps,
            "piecewise_total": sums["piecewise_total"] / num_steps,
            "l1_extreme": sums["l1_extreme"] / num_steps,
            "l2_general": sums["l2_general"] / num_steps,
            "tv_loss": sums["tv_loss"] / num_steps,
            "extreme_ratio": sums["extreme_ratio"] / num_steps,
            "enhanced_region_mean": sums["enhanced_region_mean"] / num_steps,
            "input_region_mean": sums["input_region_mean"] / num_steps,
            "avg_num_iter": sums["avg_num_iter"] / num_steps,
            "avg_input_mean": sums["avg_input_mean"] / num_steps,
            "avg_beta": sums["avg_beta"] / num_steps,
            "epoch_seconds": time.time() - epoch_start,
        }
        rows.append(row)
        print(
            "Epoch {} done | loss {:.6f} | avg_iter {:.3f}".format(
                row["epoch"],
                row["loss_total"],
                row["avg_num_iter"],
            )
        )

        epoch_path = os.path.join(config.snapshots_folder, "Epoch{:03d}.pth".format(epoch + 1))
        torch.save(DCE_net.state_dict(), epoch_path)
        torch.save(DCE_net.state_dict(), os.path.join(config.snapshots_folder, "latest.pth"))

        if best_loss is None or row["loss_total"] < best_loss:
            best_loss = row["loss_total"]
            torch.save(DCE_net.state_dict(), os.path.join(config.snapshots_folder, "best.pth"))

    with open(os.path.join(config.run_root, "train_log.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    save_json(
        os.path.join(config.run_root, "train_summary.json"),
        {
            "best_loss": best_loss,
            "final_epoch": rows[-1] if rows else None,
            "num_epochs": config.num_epochs,
            "device": str(device),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lowlight_images_path",
        type=str,
        default="/home/danqihu/danqihu/504/Zero-DiDCE_reproduction/data/lol_dataset/lol_dataset/our485/low/",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad_clip_norm", type=float, default=0.0)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--display_iter", type=int, default=20)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--target_value", type=float, default=0.6)
    parser.add_argument("--q1", type=float, default=0.2)
    parser.add_argument("--q2", type=float, default=0.8)
    parser.add_argument("--m_value", type=float, default=5.0)
    parser.add_argument("--w1", type=float, default=1.0)
    parser.add_argument("--w2", type=float, default=1.0)
    parser.add_argument("--tv_weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--load_pretrain", action="store_true")
    parser.add_argument("--pretrain_dir", type=str, default="snapshots/Epoch099.pth")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--run_root",
        type=str,
        default="/home/danqihu/danqihu/504/Zero-DiDCE_reproduction/zero_dce_paper_patch_run",
    )
    parser.add_argument(
        "--snapshots_folder",
        type=str,
        default="/home/danqihu/danqihu/504/Zero-DiDCE_reproduction/zero_dce_paper_patch_run/checkpoints/",
    )
    train(parser.parse_args())








	
