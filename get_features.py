"""Extract final-layer DINOv3 patch features for COCO, ImageNet, or NSD.

This is the single replacement for the former get_features.py and
get_features_nsd.py. It uses the unmodified upstream DINOv3 API:

    model.forward_features(images)["x_norm_patchtokens"]

By default, ViT-B/16 on 256 x 256 RGB images produces (256, 768)
per-image features, the format expected by ``DinoPro_slim``.

Example:
    python get_features.py --dataset coco --input-dir ./data/mscoco \
        --output-dir ./data/features/coco --weights ./weights/dinov3_vitb16.pth
"""

from __future__ import annotations

import argparse
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from tqdm import tqdm

# Matches the behavior of the original research scripts.
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MODEL_NAMES = (
    "dinov3_vitb16",
    "dinov3_vits16",
    "dinov3_vitl16",
    "dinov3_convnext_base",
)


@dataclass(frozen=True)
class ImageRecord:
    source: Path
    destination: Path


def _images_in(folder: Path) -> list[Path]:
    """Find image files directly in a directory (no implicit recursion)."""
    if not folder.is_dir():
        return []
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _check_requested_dirs(folders: Iterable[Path]) -> None:
    missing = [folder for folder in folders if not folder.is_dir()]
    if missing:
        raise FileNotFoundError("Requested input split(s) do not exist: " + ", ".join(map(str, missing)))


def find_coco_images(root: Path, output_dir: Path, splits: list[str] | None) -> list[ImageRecord]:
    """Read standard COCO train2014/val2014 or train2017/val2017 folders.

    COCO output is intentionally FLAT because ``COCO_feat`` searches
    ``output_dir/*.npy`` and identifies train/val by their filename stems.
    """
    if splits:
        folders = [root / split for split in splits]
        _check_requested_dirs(folders)
    else:
        folders = sorted(
            p for p in root.iterdir()
            if p.is_dir() and re.fullmatch(r"(?:train|val)\d{4}", p.name)
        )
        if not folders:
            # Also accept --input-dir pointing directly to train2014/val2014,
            # or an existing flat directory with standard COCO image names.
            folders = [root]

    records = [
        ImageRecord(source, output_dir / (source.stem + ".npy"))
        for folder in folders for source in _images_in(folder)
    ]
    unrecognized = [
        r.source for r in records
        if not ("train" in r.source.stem or "val" in r.source.stem)
    ]
    if unrecognized:
        raise ValueError(
            f"COCO filename lacks train/val marker: {unrecognized[0]}. "
            "main_prototype.py splits extracted features by filename."
        )
    return records


def find_imagenet_images(root: Path, output_dir: Path, splits: list[str] | None) -> list[ImageRecord]:
    """Read ImageNet train/<class> and val/<class> folders.

    Output is ``output_dir/<class>/<image>.npy`` to match
    ``util.dataloader.ImageNet_Feature``. Validation filenames are marked with
    'val_' because that loader uses the marker to distinguish the splits.
    """
    if root.name in {"train", "val"} and not (root / "train").exists():
        if splits and splits != [root.name]:
            raise ValueError("For a single ImageNet split directory, use --splits " + root.name)
        split_folders = [(root.name, root)]
    else:
        names = splits if splits else [s for s in ("train", "val") if (root / s).is_dir()]
        split_folders = [(name, root / name) for name in names]
        if splits:
            _check_requested_dirs(folder for _, folder in split_folders)

    if not split_folders:
        raise FileNotFoundError("Expected ImageNet train/<class>/ and/or val/<class>/ inside " + str(root))

    records = []
    for split, split_dir in split_folders:
        if split not in {"train", "val"}:
            raise ValueError("ImageNet --splits supports only 'train' and 'val'")
        classes = sorted(p for p in split_dir.iterdir() if p.is_dir())
        if not classes:
            raise ValueError(
                f"No class directories found in {split_dir}. "
                "ImageNet validation images must be organized as val/<class>/<image>.JPEG."
            )
        for class_dir in classes:
            for source in _images_in(class_dir):
                stem = source.stem
                if split == "val" and "val_" not in stem:
                    stem = "val_" + stem
                if split == "train" and "val_" in stem:
                    raise ValueError(
                        f"Training filename contains 'val_' ({source}); "
                        "the existing ImageNet feature loader would classify it as validation."
                    )
                records.append(ImageRecord(source, output_dir / class_dir.name / (stem + ".npy")))
    return records


def find_nsd_images(root: Path, output_dir: Path, splits: list[str] | None) -> list[ImageRecord]:
    """Read Algonauts 2023 preprocessed NSD training images.

    Expected: ``<root>/subjXX/training_split/training_images/*.png``.
    The original filename stem is preserved for voxel alignment.
    """
    if splits:
        raise ValueError("--splits is not used for NSD; all available subj*/training_split images are read")
    subjects = [root] if re.fullmatch(r"subj\d+", root.name) else sorted(
        p for p in root.glob("subj*") if p.is_dir()
    )
    records = []
    for subject in subjects:
        images_dir = subject / "training_split" / "training_images"
        for source in _images_in(images_dir):
            if source.suffix.lower() != ".png":
                continue
            records.append(ImageRecord(source, output_dir / subject.name / (source.stem + ".npy")))
    return records


def discover_images(
    dataset: str, root: Path, output_dir: Path, splits: list[str] | None = None
) -> list[ImageRecord]:
    root = root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {root}")

    scanners = {
        "coco": find_coco_images,
        "imagenet": find_imagenet_images,
        "nsd": find_nsd_images,
    }
    records = scanners[dataset](root, output_dir, splits)
    if not records:
        raise FileNotFoundError(
            f"No {dataset} images were found under {root}. "
            "Check the documented dataset directory layout."
        )

    # Prevent silent overwriting if two source images resolve to the same path.
    destinations: set[Path] = set()
    for record in records:
        if record.destination in destinations:
            raise ValueError(f"Conflicting images share an output filename: {record.destination}")
        destinations.add(record.destination)
    return records


def make_transform(image_size: int) -> v2.Compose:
    """The same RGB resize and ImageNet normalization as the input scripts."""
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((image_size, image_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


class FeatureImageDataset(Dataset):
    def __init__(self, records: list[ImageRecord], image_size: int):
        self.records = records
        self.transform = make_transform(image_size)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path = self.records[index].source
        try:
            with Image.open(path) as image:
                rgb = image.convert("RGB")
            return self.transform(rgb), index
        except Exception as exc:
            raise RuntimeError(f"Cannot load image {path}") from exc


def load_dinov3(name: str, weights: str, device: torch.device) -> torch.nn.Module:
    """Load official DINOv3 weights without editing its model source."""
    try:
        from dinov3.hub import backbones
    except ImportError as exc:
        raise ImportError(
            "DINOv3 is not installed. Install this repository's requirements.txt "
            "(including dinov3) before feature extraction."
        ) from exc

    # Official constructors accept a local checkpoint path or a URL.
    if not weights.startswith(("https://", "http://", "file://")):
        path = Path(weights).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"DINOv3 weights not found: {path}")
        weights = str(path)
    constructor = getattr(backbones, name)
    return constructor(weights=weights).to(device).eval()


def save_numpy_atomic(path: Path, values: np.ndarray) -> None:
    """Do not leave partial .npy files after interrupted processing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix="." + path.stem + "-", suffix=".npy", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.save(handle, values)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def extract(args: argparse.Namespace) -> tuple[int, int]:
    records = discover_images(args.dataset, args.input_dir, args.output_dir, args.splits)
    if args.max_images is not None:
        records = records[: args.max_images]
    print(f"Found {len(records):,} {args.dataset} images")
    if args.dry_run:
        for record in records[:5]:
            print(f"  {record.source} -> {record.destination}")
        print("Dry run: no model loaded or features written.")
        return 0, 0

    if not args.weights:
        raise ValueError("--weights is required (path or authorized URL to the DINOv3 checkpoint)")
    already_done = sum(record.destination.is_file() for record in records)
    if not args.overwrite:
        records = [record for record in records if not record.destination.is_file()]
    if not records:
        print(f"All outputs already exist ({already_done:,} skipped). Use --overwrite to regenerate.")
        return 0, already_done
    if not args.overwrite and already_done:
        print(f"Skipping {already_done:,} existing feature files (use --overwrite to replace them).")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available. Use --device cpu or --device auto.")
    if args.model != "dinov3_vitb16":
        print("Note: non-default DINOv3 models may produce dimensions incompatible with DinoPro_slim.")
    if args.image_size != 256:
        print("Note: the default ProtoMRI prototype model requires 256 x 256 extraction inputs.")

    model = load_dinov3(args.model, args.weights, device)
    dataset = FeatureImageDataset(records, args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    saved = 0
    out_dtype = np.float16 if args.save_dtype == "float16" else np.float32
    printed_shape = False

    with torch.inference_mode():
        for images, indices in tqdm(loader, desc="Extracting patch tokens", unit="batch"):
            images = images.to(device, non_blocking=(device.type == "cuda"))
            feature_output = model.forward_features(images)
            if not isinstance(feature_output, dict) or "x_norm_patchtokens" not in feature_output:
                raise TypeError(
                    "DINOv3 forward_features() did not return 'x_norm_patchtokens'. "
                    "Check the installed upstream model version."
                )
            tokens = feature_output["x_norm_patchtokens"]
            if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
                raise ValueError(f"Expected patch-token tensor (B, N, C), got {type(tokens)}")
            if args.model == "dinov3_vitb16" and args.image_size == 256:
                if tokens.shape[1:] != (256, 768):
                    raise ValueError(f"Expected ViT-B/16 shape (B, 256, 768), got {tuple(tokens.shape)}")
            if not printed_shape:
                print(f"Saving {tuple(tokens.shape[1:])} patch tokens per image as {args.save_dtype}")
                printed_shape = True

            array = tokens.detach().to(device="cpu", dtype=torch.float32).numpy()
            for i, index in enumerate(indices.tolist()):
                target = records[index].destination
                save_numpy_atomic(target, array[i].astype(out_dtype, copy=False))
                saved += 1

    print(f"Saved {saved:,} feature arrays under {args.output_dir}")
    return saved, already_done if not args.overwrite else 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract last-layer DINOv3 patch tokens for COCO, ImageNet, or Algonauts 2023 NSD"
    )
    parser.add_argument("--dataset", required=True, choices=("coco", "imagenet", "nsd"))
    parser.add_argument("--input-dir", type=Path, required=True, help="Root directory of source images")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for .npy features")
    parser.add_argument("--weights", type=str, default=None, help="Local DINOv3 weights path or authorized URL")
    parser.add_argument("--model", choices=MODEL_NAMES, default="dinov3_vitb16")
    parser.add_argument("--image-size", type=int, default=256, help="Square image resolution (default: 256)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--save-dtype", default="float32", choices=("float32", "float16"))
    parser.add_argument(
        "--splits", nargs="+", default=None,
        help="Optional subfolders: COCO train2014 val2014; ImageNet train val; not used for NSD",
    )
    parser.add_argument("--overwrite", action="store_true", help="Recompute existing .npy feature files")
    parser.add_argument("--max-images", type=int, default=None, help="Limit images (debugging)")
    parser.add_argument("--dry-run", action="store_true", help="Show input/output mapping without inference")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("--batch-size must be >= 1 and --num-workers must be >= 0")
    if args.image_size < 16 or args.image_size % 16 != 0:
        parser.error("--image-size must be a positive multiple of 16")
    if args.max_images is not None and args.max_images < 1:
        parser.error("--max-images must be >= 1")
    return args


if __name__ == "__main__":
    extract(parse_args())
