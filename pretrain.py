"""Pretrain LeJEPA on ImageNet-100 with an online linear probe."""

import argparse
import math
import os
import sys
import time

import timm
import torch
import torch.nn.functional as F
from lightning.fabric import Fabric
from lightning.fabric.loggers import TensorBoardLogger
from torch import nn
from torch.optim.lr_scheduler import LambdaLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm

from datasets_utils import im100_ssl_dataset
from jepa_models import LeJEPA
from sigreg import SIGReg
from utils import Tee, format_duration, save_args


def parse_args() -> argparse.Namespace:
    """Parse the dataset, optimization, and output settings."""
    parser = argparse.ArgumentParser(description="Pretraining")
    parser.add_argument(
        "--dataset", type=str, default="im100", choices=["im100"]
    )
    parser.add_argument(
        "--data_path", type=str, default="/data/datasets/ImageNet-100"
    )
    parser.add_argument(
        "--vision_model", type=str, default="vit_small_patch16_224"
    )
    parser.add_argument("--num_views", type=int, default=2)
    parser.add_argument("--proj_dim", type=int, default=64)
    parser.add_argument("--probe_num_classes", type=int, default=100)
    parser.add_argument("--devices", type=int, default=4)
    parser.add_argument(
        "--strategy", default="ddp", choices=["auto", "ddp"]
    )
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--wd", type=float, default=5e-2)
    parser.add_argument(
        "--bs", type=int, default=128, help="Batch size per GPU."
    )
    parser.add_argument("--lamb", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/LeJEPA_vit_small_patch16_224_V2",
    )
    args = parser.parse_args()
    return args


def main():
    start_time = time.perf_counter()

    # Parse arguments
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # Set up logging and Fabric
    tb_logger = TensorBoardLogger(
        root_dir=os.path.join(args.save_dir, "logs"), name="tb_logs"
    )
    fabric = Fabric(
        accelerator="gpu",
        strategy=args.strategy,
        devices=args.devices,
        precision="bf16-mixed",
        loggers=[tb_logger],
    )
    fabric.launch()

    # Write the log and arguments files only on rank zero.
    if fabric.is_global_zero:
        sys.stdout = Tee(os.path.join(args.save_dir, "log.log"))
        save_args(args, args.save_dir)

    # Seed for reproducibility
    fabric.seed_everything(args.seed)

    # Load the dataset
    train_dataset, valid_dataset = im100_ssl_dataset(
        root=args.data_path, num_views=args.num_views
    )
    train_dl = DataLoader(
        train_dataset,
        batch_size=args.bs,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    valid_dl = DataLoader(
        valid_dataset,
        batch_size=args.bs,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    train_dl, valid_dl = fabric.setup_dataloaders(train_dl, valid_dl)

    # Create the vision encoder without a classification head.
    if "resnet" in args.vision_model:
        vision_encoder = timm.create_model(
            args.vision_model,
            pretrained=False,
            num_classes=0,
            norm_layer=nn.SyncBatchNorm,
        )
    elif "vit" in args.vision_model:
        vision_encoder = timm.create_model(
            args.vision_model,
            pretrained=False,
            num_classes=0,
            global_pool="token",
            drop_path_rate=0.1,
        )
    else:
        raise ValueError(
            f"Unsupported vision model: {args.vision_model!r}."
        )

    # Construct LeJEPA with the configured SIGReg regularizer.
    sigreg = SIGReg(
        knots=17,
        num_proj=1024,
        distributed=fabric.world_size > 1,
    )
    ssl_model = LeJEPA(
        vision_encoder=vision_encoder,
        sigreg=sigreg,
        lamb=args.lamb,
        proj_dim=args.proj_dim,
    )
    ssl_model = fabric.setup_module(ssl_model)

    # Probe for linear evaluation on detached features.
    probe = nn.Sequential(
        nn.LayerNorm(vision_encoder.num_features),
        nn.Linear(vision_encoder.num_features, args.probe_num_classes),
    )
    probe = fabric.setup_module(probe)

    # Define parameter groups and optimizer
    ssl_group = {
        "params": ssl_model.parameters(),
        "lr": args.lr,
        "weight_decay": args.wd,
    }
    probe_group = {
        "params": probe.parameters(),
        "lr": 1e-3,
        "weight_decay": 1e-7,
    }
    optimizer = torch.optim.AdamW([ssl_group, probe_group])

    # Defining learning rate scheduler
    warmup_steps = args.warmup_epochs * len(train_dl)
    total_steps = args.epochs * len(train_dl)
    s1 = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    def ssl_lr_factor(step):
        """Cosine decay from the SSL base rate to one thousandth of it."""
        progress = step / (total_steps - warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return 0.001 + 0.999 * cosine
    s2 = LambdaLR(
        optimizer, lr_lambda=[ssl_lr_factor, lambda step: 1.0]
    )
    scheduler = SequentialLR(
        optimizer, schedulers=[s1, s2], milestones=[warmup_steps]
    )

    # Setup optimizer with Fabric
    optimizer = fabric.setup_optimizers(optimizer)

    # Training loop
    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        fabric.print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_step(
            ssl_model,
            probe,
            train_dl,
            optimizer,
            scheduler,
            fabric,
            num_classes=args.probe_num_classes,
        )
        validation_step(
            ssl_model,
            probe,
            valid_dl,
            fabric,
            step=scheduler.last_epoch,
            num_classes=args.probe_num_classes,
        )
        if (epoch + 1) % 50 == 0 or epoch == args.epochs - 1:
            # Save each SSL submodule's weights under its own name.
            fabric.save(
                os.path.join(args.save_dir, f"model_epoch_{epoch}.pth"),
                dict(ssl_model.module.named_children()),
            )
        now = time.perf_counter()
        fabric.print(
            f"Time: epoch={format_duration(now - epoch_start)} | "
            f"elapsed={format_duration(now - start_time)}"
        )

    fabric.barrier()  # Wait for all ranks to finish training
    fabric.print(
        f"\nTotal time: {format_duration(time.perf_counter() - start_time)}"
    )
    fabric.print("\nEND")

    return None


def train_step(
    ssl_model,
    probe,
    train_dl,
    optimizer,
    scheduler,
    fabric,
    *,
    num_classes,
):
    """Training step with batch progress and returned epoch metrics."""
    ssl_model.train()
    probe.train()

    loss_means = {}
    # Micro averaging gives top-1 accuracy over all samples.
    accuracy = MulticlassAccuracy(
        num_classes=num_classes, average="micro", top_k=1
    ).to(fabric.device)

    with tqdm(
        train_dl,
        desc="Training",
        leave=False,
        dynamic_ncols=True,
        disable=not fabric.is_global_zero,
    ) as progress:
        for views, labels in progress:
            optimizer.zero_grad(set_to_none=True)
            losses, features = ssl_model(views, get_features=True)
            logits = probe(features.detach())
            targets = labels.repeat_interleave(views.shape[1])
            probe_loss = F.cross_entropy(logits.float(), targets)
            loss = losses["ssl_loss"] + probe_loss
            fabric.backward(loss)
            optimizer.step()

            batch_losses = {
                name: value.detach() for name, value in losses.items()
            }
            batch_losses["probe_loss"] = probe_loss.detach()
            for name, value in batch_losses.items():
                if name not in loss_means:
                    loss_means[name] = MeanMetric().to(fabric.device)
                # The probe classifies each view separately.
                weight = (
                    targets.numel() if name == "probe_loss" else labels.size(0)
                )
                loss_means[name].update(value, weight=weight)

            accuracy.update(logits.detach().argmax(dim=1), targets)
            metrics = {
                f"train/{name}": value for name, value in batch_losses.items()
            }
            fabric.log_dict(metrics, step=scheduler.last_epoch)
            scheduler.step()

            if fabric.is_global_zero:
                postfix = {
                    name: f"{value.item():.4g}"
                    for name, value in batch_losses.items()
                }
                progress.set_postfix(postfix)

    epoch_losses = {
        name: metric.compute().item() for name, metric in loss_means.items()
    }
    epoch_accuracy = 100.0 * accuracy.compute().item()
    metrics = {
        f"train_epoch/{name}": value for name, value in epoch_losses.items()
    }
    metrics["train_epoch/acc_top1_micro"] = epoch_accuracy
    fabric.log_dict(metrics, step=scheduler.last_epoch)
    summary = " | ".join(
        f"{name}={value:.4g}" for name, value in epoch_losses.items()
    )
    fabric.print(f"Training: {summary} | acc_top1_micro={epoch_accuracy:.2f}%")
    return metrics


@torch.inference_mode()
def validation_step(ssl_model, probe, valid_dl, fabric, step, num_classes):
    """Validation step."""
    ssl_model.eval()
    probe.eval()

    loss_mean = MeanMetric().to(fabric.device)
    # Micro averaging gives top-1 accuracy over all samples.
    accuracy = MulticlassAccuracy(
        num_classes=num_classes, average="micro", top_k=1
    ).to(fabric.device)
    for images, labels in valid_dl:
        with fabric.autocast():
            features = ssl_model.encoder(images)
            logits = probe(features)
        probe_loss = F.cross_entropy(logits.float(), labels)
        loss_mean.update(probe_loss, weight=labels.size(0))
        accuracy.update(logits.argmax(dim=1), labels)

    metrics = {
        "val/probe_loss": loss_mean.compute().item(),
        "val/acc_top1_micro": 100.0 * accuracy.compute().item(),
    }
    fabric.log_dict(metrics, step=step)
    fabric.print(
        f"Validation: probe_loss={metrics['val/probe_loss']:.4g} | "
        f"acc_top1_micro={metrics['val/acc_top1_micro']:.2f}%"
    )
    return metrics


if __name__ == "__main__":
    main()
