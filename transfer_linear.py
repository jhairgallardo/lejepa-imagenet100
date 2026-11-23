"""Linear evaluation of a frozen encoder, following AugSelf and CASSLE.

A linear classifier is fit with L-BFGS on cached features. The L2 weight is
chosen on the validation split, the selected classifier is refit on the
training and validation splits together, and its test accuracy is reported.
"""

import argparse
import os
import sys
import time
from copy import deepcopy

import timm
import torch
import torch.nn.functional as F
from lightning.fabric import Fabric
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets_utils import load_linear_dataset
from utils import Tee, format_duration, save_args

# AugSelf and CASSLE sweep 45 log-spaced L2 weights with these L-BFGS options.
L2_WEIGHTS = torch.logspace(-6, 5, steps=45).tolist()
LBFGS_KWARGS = {
    "lr": 1.0,
    "max_iter": 5000,
    "line_search_fn": "strong_wolfe",
    "tolerance_grad": 1e-10,
    "tolerance_change": 0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Linear Eval")
    parser.add_argument(
        "--dataset",
        type=str,
        default="im100",
        choices=[
            "im100", "cifar10", "cifar100", "food101", "pets", "flowers102",
            "caltech101", "cars", "aircraft", "dtd", "sun397", "mit67",
            "cub200",
        ],
    )
    parser.add_argument(
        "--data_path", type=str, default="/data/datasets/ImageNet-100"
    )
    parser.add_argument("--pretrained_encoder", type=str, required=True)
    parser.add_argument("--vision_model", type=str, default="resnet50")
    parser.add_argument(
        "--metric",
        type=str,
        required=True,
        choices=["micro", "macro"],
        help="Top-1 accuracy used on the validation split to choose the L2 "
        "weight: micro (top-1) or macro (mean per-class).",
    )
    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="results/linear")
    args = parser.parse_args()
    return args


def main():
    start_time = time.perf_counter()
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # Configure logging and single-GPU float32 evaluation.
    fabric = Fabric(
        accelerator="gpu",
        strategy="auto",
        devices=1,
        precision="32-true",
    )
    fabric.launch()
    sys.stdout = Tee(os.path.join(args.save_dir, "log.log"))
    save_args(args, args.save_dir)

    # Load images once, without shuffling or dropping samples.
    image_datasets = load_linear_dataset(
        name=args.dataset,
        root=args.data_path,
    )
    image_loaders = [
        DataLoader(
            dataset,
            batch_size=args.bs,
            shuffle=False,
            drop_last=False,
            num_workers=args.workers,
            pin_memory=True,
        )
        for dataset in image_datasets
    ]

    # Load the frozen encoder and cache its features on the GPU.
    encoder = load_pretrained_encoder(
        args.pretrained_encoder, args.vision_model
    ).to(fabric.device)
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = [
        extract_features(encoder, loader, fabric, split)
        for loader, split in zip(image_loaders, ("train", "val", "test"))
    ]
    del image_loaders, image_datasets, encoder
    torch.cuda.empty_cache()

    feature_dim = train_x.shape[1]
    num_classes = int(train_y.max().item()) + 1
    fabric.print(
        f"Features: train={len(train_y)} | val={len(val_y)} | "
        f"test={len(test_y)} | dim={feature_dim} | classes={num_classes}"
    )
    fabric.print(
        f"Choosing the L2 weight by validation acc_top1_{args.metric}."
    )

    # Sweep the L2 weight. Each fit starts from the previous solution.
    classifier = nn.Linear(feature_dim, num_classes).to(fabric.device)
    best_accuracy = -1.0
    for weight in L2_WEIGHTS:
        fit_lbfgs(classifier, train_x, train_y, weight)
        val_metrics = evaluate(classifier, val_x, val_y)
        fabric.print(f"w={weight:.3e} | val: {format_metrics(val_metrics)}")

        if val_metrics[f"acc_top1_{args.metric}"] > best_accuracy:
            best_accuracy = val_metrics[f"acc_top1_{args.metric}"]
            best_weight = weight
            best_val_metrics = val_metrics
            best_classifier = deepcopy(classifier)

    fabric.print(
        f"\nBest L2 weight: w={best_weight:.3e} | "
        f"val: {format_metrics(best_val_metrics)}"
    )
    train_only_metrics = evaluate(best_classifier, test_x, test_y)
    fabric.print(f"Test (train only): {format_metrics(train_only_metrics)}")

    # Refit the selected classifier on train+val and evaluate it on test.
    fit_lbfgs(
        best_classifier,
        torch.cat([train_x, val_x]),
        torch.cat([train_y, val_y]),
        best_weight,
    )
    test_metrics = evaluate(best_classifier, test_x, test_y)
    fabric.print(f"Test (train+val): {format_metrics(test_metrics)}")

    fabric.save(
        os.path.join(args.save_dir, "classifier.pt"),
        {
            "classifier": best_classifier,
            "l2_weight": best_weight,
            "val_metrics": best_val_metrics,
            "test_metrics": test_metrics,
            "args": vars(args),
        },
    )
    report_key = f"acc_top1_{args.metric}"
    fabric.print(
        f"\nReport: {report_key}={test_metrics[report_key]:.2f}% "
        f"(top-1 {args.metric}, test after the train+val refit)"
    )
    fabric.print(
        f"\nTotal time: {format_duration(time.perf_counter() - start_time)}"
    )


def fit_lbfgs(classifier, features, labels, weight):
    """Minimize summed cross-entropy plus an L2 penalty with L-BFGS."""
    optimizer = torch.optim.LBFGS(classifier.parameters(), **LBFGS_KWARGS)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(classifier(features), labels, reduction="sum")
        # AugSelf and CASSLE penalize both the weights and the bias.
        for parameter in classifier.parameters():
            loss = loss + weight * parameter.pow(2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)


@torch.no_grad()
def evaluate(classifier, features, labels):
    """Return the mean loss and top-1 micro and macro accuracy in percent."""
    logits = classifier(features)
    correct = (logits.argmax(dim=1) == labels).float()

    # Macro accuracy averages per-class accuracy over the classes present.
    class_correct = torch.bincount(labels, weights=correct)
    class_counts = torch.bincount(labels)
    present = class_counts > 0
    class_accuracy = class_correct[present] / class_counts[present]

    return {
        "loss": F.cross_entropy(logits, labels).item(),
        "acc_top1_micro": 100.0 * correct.mean().item(),
        "acc_top1_macro": 100.0 * class_accuracy.mean().item(),
    }


def format_metrics(metrics):
    """Format evaluation metrics for the log."""
    return (
        f"loss={metrics['loss']:.4g} | "
        f"acc_top1_micro={metrics['acc_top1_micro']:.2f}% | "
        f"acc_top1_macro={metrics['acc_top1_macro']:.2f}%"
    )


class ViTLastTwoCLS(nn.Module):
    """Concatenate the normalized CLS tokens of a ViT's last two blocks.

    This follows LeJEPA's linear-probe protocol, which uses DINO's
    convention of applying the final norm to each block's output.
    """

    def __init__(self, vit):
        super().__init__()
        self.vit = vit

    def forward(self, x):
        layers = self.vit.forward_intermediates(
            x,
            indices=2,
            norm=True,
            return_prefix_tokens=True,
            output_fmt="NLC",
            intermediates_only=True,
        )
        # Prefix token 0 is the CLS token; the last block comes second.
        return torch.cat([prefix[:, 0] for _, prefix in layers], dim=-1)


def load_pretrained_encoder(checkpoint_path, vision_model):
    """Load and freeze the pretrained encoder.

    ResNets return pooled features. ViTs return the concatenated CLS
    tokens of their last two blocks (see ``ViTLastTwoCLS``).
    """
    if "resnet" in vision_model:
        encoder = timm.create_model(
            vision_model,
            pretrained=False,
            num_classes=0,
        )
    elif "vit" in vision_model:
        encoder = timm.create_model(
            vision_model,
            pretrained=False,
            num_classes=0,
            global_pool="token",
        )
    else:
        raise ValueError(f"Unsupported vision model: {vision_model!r}.")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    encoder.load_state_dict(checkpoint["encoder"], strict=True)
    if "vit" in vision_model:
        assert encoder.has_class_token, "ViT probing needs a CLS token."
        encoder = ViTLastTwoCLS(encoder)
    encoder.requires_grad_(False)
    encoder.eval()
    return encoder


@torch.no_grad()
def extract_features(encoder, loader, fabric, split):
    """Return float32 features and labels for one split on the GPU."""
    encoder.eval()
    feature_batches = []
    label_batches = []

    for images, labels in tqdm(
        loader, desc=f"Extracting {split}", leave=False
    ):
        # CUB supplies multiple crops for each training/validation image.
        if images.ndim == 5:
            num_crops = images.shape[1]
            images = images.flatten(0, 1)
            labels = labels.repeat_interleave(num_crops)

        # Keep the encoder batch size bounded when using multiple crops.
        for image_batch in images.split(loader.batch_size):
            image_batch = image_batch.to(
                fabric.device, non_blocking=True
            )
            feature_batches.append(encoder(image_batch).float().cpu())

        label_batches.append(labels.cpu())

    features = torch.cat(feature_batches).to(fabric.device)
    labels = torch.cat(label_batches).to(fabric.device)
    return features, labels


if __name__ == "__main__":
    main()
