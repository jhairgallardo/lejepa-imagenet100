"""SSL and linear evaluation datasets and augmentations."""

import os
from pathlib import Path

import torch
from PIL import Image, ImageFilter
from torch.utils.data import Subset, random_split
from torchvision.datasets import (
    CIFAR10,
    CIFAR100,
    DTD,
    SUN397,
    Caltech101,
    FGVCAircraft,
    Flowers102,
    Food101,
    ImageFolder,
    OxfordIIITPet,
    StanfordCars,
)
from torchvision.transforms import v2 as transforms

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

transform_for_linear = transforms.Compose(
    [
        transforms.Resize(
            224, interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.CenterCrop(224),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=MEAN, std=STD),
    ]
)


class MultiView:
    """Stack independent augmented views of one image."""

    def __init__(self, transform, num_views):
        self.transform = transform
        self.num_views = num_views

    def __call__(self, image):
        # Independent calls, not copies of the same crop.
        return torch.stack(
            [self.transform(image) for _ in range(self.num_views)]
        )


class PILGaussianBlur:
    """Blur a PIL image with a uniformly sampled radius."""

    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma

    def __call__(self, image):
        sigma = torch.empty(1).uniform_(*self.sigma).item()
        return image.filter(ImageFilter.GaussianBlur(radius=sigma))


def im100_ssl_dataset(root, num_views):
    """Return ImageNet-100 SSL datasets in the order (train, val).

    Each training sample stacks ``num_views`` independent augmented views.
    """
    view_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                224,
                scale=(0.2, 1.0),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([PILGaussianBlur(sigma=(0.1, 2.0))], p=0.5),
            transforms.RandomSolarize(threshold=128, p=0.2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )
    train_transform = MultiView(view_transform, num_views)
    valid_transform = transforms.Compose(
        [
            transforms.Resize(
                224, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.CenterCrop(224),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )

    train_dataset = ImageFolder(
        root=os.path.join(root, "train"), transform=train_transform
    )
    valid_dataset = ImageFolder(
        root=os.path.join(root, "val"), transform=valid_transform
    )
    return train_dataset, valid_dataset


def im100_linear_dataset(root):
    """Return ImageNet-100 datasets in the order (train, val, test).

    The validation split is a random 10% of the training set, and the test
    set is the official ImageNet validation set.
    """
    trainval_dataset = ImageFolder(
        root=os.path.join(root, "train"), transform=transform_for_linear
    )
    test_dataset = ImageFolder(
        root=os.path.join(root, "val"), transform=transform_for_linear
    )
    num_train = int(0.9 * len(trainval_dataset))
    num_val = len(trainval_dataset) - num_train
    train_dataset, val_dataset = random_split(
        trainval_dataset,
        [num_train, num_val],
        generator=torch.Generator().manual_seed(42),
    )

    return train_dataset, val_dataset, test_dataset


def cifar10_linear_dataset(root):
    """Return CIFAR-10 datasets in the order (train, val, test)."""
    trainval_dataset = CIFAR10(
        root=root,
        train=True,
        transform=transform_for_linear,
    )
    train_dataset, val_dataset = random_split(
        trainval_dataset,
        [45000, 5000],
        generator=torch.Generator().manual_seed(43),
    )
    test_dataset = CIFAR10(
        root=root,
        train=False,
        transform=transform_for_linear,
    )

    return train_dataset, val_dataset, test_dataset


def cifar100_linear_dataset(root):
    """Return CIFAR-100 datasets in the order (train, val, test)."""
    trainval_dataset = CIFAR100(
        root=root,
        train=True,
        transform=transform_for_linear,
    )
    train_dataset, val_dataset = random_split(
        trainval_dataset,
        [45000, 5000],
        generator=torch.Generator().manual_seed(44),
    )
    test_dataset = CIFAR100(
        root=root,
        train=False,
        transform=transform_for_linear,
    )

    return train_dataset, val_dataset, test_dataset


def food101_linear_dataset(root):
    """Return Food-101 datasets in the order (train, val, test)."""
    trainval_dataset = Food101(
        root=root,
        split="train",
        transform=transform_for_linear,
    )
    train_dataset, val_dataset = random_split(
        trainval_dataset,
        [68175, 7575],
        generator=torch.Generator().manual_seed(42),
    )
    test_dataset = Food101(
        root=root,
        split="test",
        transform=transform_for_linear,
    )

    return train_dataset, val_dataset, test_dataset


def pets_linear_dataset(root):
    """Return Oxford-IIIT Pet datasets in the order (train, val, test)."""
    # https://huggingface.co/zhengli97/prompt_learning_dataset/blob/main/oxford_pets.zip
    trainval_dataset = OxfordIIITPet(
        root=root,
        split="trainval",
        transform=transform_for_linear,
    )
    train_dataset, val_dataset = random_split(
        trainval_dataset,
        [2940, 740],
        generator=torch.Generator().manual_seed(49),
    )
    test_dataset = OxfordIIITPet(
        root=root,
        split="test",
        transform=transform_for_linear,
    )

    return train_dataset, val_dataset, test_dataset


def flowers102_linear_dataset(root):
    """Return Flowers-102 datasets in the order (train, val, test)."""
    # https://huggingface.co/zhengli97/prompt_learning_dataset/blob/main/oxford_flowers.zip
    # https://huggingface.co/datasets/cnak47/cv-datasets/resolve/main/setid.mat
    train_dataset = Flowers102(
        root=root,
        split="train",
        transform=transform_for_linear,
    )
    val_dataset = Flowers102(
        root=root,
        split="val",
        transform=transform_for_linear,
    )
    test_dataset = Flowers102(
        root=root,
        split="test",
        transform=transform_for_linear,
    )

    return train_dataset, val_dataset, test_dataset


def caltech101_linear_dataset(root):
    """Return Caltech101 datasets in the order (train, val, test)."""
    # https://huggingface.co/zhengli97/prompt_learning_dataset/blob/main/caltech-101.zip
    transform = transforms.Compose([transforms.RGB(), transform_for_linear])
    dataset = Caltech101(root=root, transform=transform)
    split_path = Path(__file__).resolve().parent / "splits" / "caltech101.pth"
    train_indices, val_indices, test_indices = torch.load(
        split_path, map_location="cpu", weights_only=True
    )
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, val_dataset, test_dataset


def cars_linear_dataset(root):
    """Return Stanford Cars datasets in the order (train, val, test)."""
    # https://huggingface.co/zhengli97/prompt_learning_dataset/blob/main/stanford_cars.zip
    # Torchvision appends "stanford_cars" to its root.
    root = Path(root).resolve().parent
    trainval_dataset = StanfordCars(
        root=root,
        split="train",
        transform=transform_for_linear,
    )
    train_dataset, val_dataset = random_split(
        trainval_dataset,
        [7000, 1144],
        generator=torch.Generator().manual_seed(51),
    )
    test_dataset = StanfordCars(
        root=root,
        split="test",
        transform=transform_for_linear,
    )

    return train_dataset, val_dataset, test_dataset


def aircraft_linear_dataset(root):
    """Return FGVC-Aircraft datasets in the order (train, val, test)."""
    # https://huggingface.co/zhengli97/prompt_learning_dataset/blob/main/fgvc_aircraft.zip
    train_dataset = FGVCAircraft(
        root=root,
        split="train",
        annotation_level="variant",
        transform=transform_for_linear,
    )
    val_dataset = FGVCAircraft(
        root=root,
        split="val",
        annotation_level="variant",
        transform=transform_for_linear,
    )
    test_dataset = FGVCAircraft(
        root=root,
        split="test",
        annotation_level="variant",
        transform=transform_for_linear,
    )

    return train_dataset, val_dataset, test_dataset


def dtd_linear_dataset(root):
    """Return DTD datasets in the order (train, val, test)."""
    # https://huggingface.co/zhengli97/prompt_learning_dataset/blob/main/dtd.zip
    train_dataset = DTD(
        root=root,
        split="train",
        partition=1,
        transform=transform_for_linear,
    )
    val_dataset = DTD(
        root=root,
        split="val",
        partition=1,
        transform=transform_for_linear,
    )
    test_dataset = DTD(
        root=root,
        split="test",
        partition=1,
        transform=transform_for_linear,
    )

    return train_dataset, val_dataset, test_dataset


def sun397_linear_dataset(root):
    """Return SUN397 datasets in the order (train, val, test)."""
    # https://huggingface.co/zhengli97/prompt_learning_dataset/blob/main/sun397.zip
    root = Path(root)
    dataset = SUN397(root=root, transform=transform_for_linear)
    path_to_index = {
        path.relative_to(root / "SUN397").as_posix(): index
        for index, path in enumerate(dataset._image_files)
    }

    with open(root / "Training_01.txt") as file:
        trainval_indices = [
            path_to_index[line.strip().lstrip("/")] for line in file
        ]
    with open(root / "Testing_01.txt") as file:
        test_indices = [
            path_to_index[line.strip().lstrip("/")] for line in file
        ]
    trainval_dataset = Subset(dataset, trainval_indices)

    split_path = Path(__file__).resolve().parent / "splits" / "sun397.pth"
    train_indices, val_indices = torch.load(
        split_path, map_location="cpu", weights_only=True
    )
    train_dataset = Subset(trainval_dataset, train_indices)
    val_dataset = Subset(trainval_dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, val_dataset, test_dataset


def mit67_linear_dataset(root):
    """Return MIT67 datasets in the order (train, val, test)."""
    # Images: https://huggingface.co/datasets/zhangyun04/VLM-UB/tree/main/mit_indoor67
    # Train split: https://web.mit.edu/torralba/www/TrainImages.txt
    # Test split: https://web.mit.edu/torralba/www/TestImages.txt
    root = Path(root)
    image_root = root / "Images"
    dataset = ImageFolder(root=image_root, transform=transform_for_linear)
    with open(root / "TrainImages.txt") as file:
        train_paths = {line.strip() for line in file}
    with open(root / "TestImages.txt") as file:
        test_paths = {line.strip() for line in file}

    trainval_indices = []
    test_indices = []
    for index, (path, _) in enumerate(dataset.samples):
        relative_path = Path(path).relative_to(image_root).as_posix()
        if relative_path in train_paths:
            trainval_indices.append(index)
        if relative_path in test_paths:
            test_indices.append(index)

    trainval_dataset = Subset(dataset, trainval_indices)
    train_dataset, val_dataset = random_split(
        trainval_dataset,
        [4690, 670],
        generator=torch.Generator().manual_seed(51),
    )
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, val_dataset, test_dataset


def cub200_linear_dataset(root):
    """Return CUB-200-2011 datasets in the order (train, val, test)."""
    # https://data.caltech.edu/records/65de6-vp158
    train_transform = transforms.Compose(
        [
            transforms.Resize(
                256, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.TenCrop(224),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=MEAN, std=STD),
            torch.stack,
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(
                256, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.CenterCrop(224),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )

    root = Path(root)
    image_root = root / "images"
    trainval_source = ImageFolder(root=image_root, transform=train_transform)
    test_source = ImageFolder(root=image_root, transform=test_transform)
    with open(root / "images.txt") as file:
        image_paths = dict(line.strip().split(maxsplit=1) for line in file)
    with open(root / "train_test_split.txt") as file:
        is_training = {}
        for line in file:
            image_id, flag = line.split()
            is_training[image_paths[image_id]] = flag == "1"

    trainval_indices = []
    test_indices = []
    for index, (path, _) in enumerate(trainval_source.samples):
        # The saved CASSLE split matches the RGB-only ImageFolder ordering.
        with Image.open(path) as image:
            if image.mode != "RGB":
                continue
        relative_path = Path(path).relative_to(image_root).as_posix()
        if is_training[relative_path]:
            trainval_indices.append(index)
        else:
            test_indices.append(index)

    trainval_dataset = Subset(trainval_source, trainval_indices)
    split_path = Path(__file__).resolve().parent / "splits" / "cub200.pth"
    train_indices, val_indices = torch.load(
        split_path, map_location="cpu", weights_only=True
    )
    train_dataset = Subset(trainval_dataset, train_indices)
    val_dataset = Subset(trainval_dataset, val_indices)
    test_dataset = Subset(test_source, test_indices)

    return train_dataset, val_dataset, test_dataset


def load_linear_dataset(name, root):
    """Return the selected linear dataset's (train, val, test) splits."""
    dataset_functions = {
        "im100": im100_linear_dataset,
        "cifar10": cifar10_linear_dataset,
        "cifar100": cifar100_linear_dataset,
        "food101": food101_linear_dataset,
        "pets": pets_linear_dataset,
        "flowers102": flowers102_linear_dataset,
        "caltech101": caltech101_linear_dataset,
        "cars": cars_linear_dataset,
        "aircraft": aircraft_linear_dataset,
        "dtd": dtd_linear_dataset,
        "sun397": sun397_linear_dataset,
        "mit67": mit67_linear_dataset,
        "cub200": cub200_linear_dataset,
    }
    if name not in dataset_functions:
        supported = ", ".join(dataset_functions)
        raise ValueError(
            f"Unknown linear dataset {name!r}. Choose from: {supported}."
        )
    return dataset_functions[name](root=root)
