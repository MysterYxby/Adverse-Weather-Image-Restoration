#!/usr/bin/env python
"""
Compute common image restoration metrics.

Core metrics:
  - PSNR: no extra dependency beyond numpy/Pillow
  - SSIM: uses scikit-image

Optional metrics:
  - LPIPS: pip install lpips torch torchvision
  - NIQE/BRISQUE: pip install pyiqa torch torchvision
  - FID: pip install clean-fid

Examples:
  python compute_metrics.py --pred results --gt gt --metrics psnr ssim --output metrics.csv
  python compute_metrics.py --pred results --gt gt --metrics psnr ssim lpips --device cuda
  python compute_metrics.py --pred results --metrics niqe brisque --output no_ref_metrics.csv
  python compute_metrics.py --pred results --gt gt --metrics fid
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
REFERENCE_METRICS = {"psnr", "ssim", "lpips"}
NO_REFERENCE_METRICS = {"niqe", "brisque"}
DATASET_METRICS = {"fid"}
SUPPORTED_METRICS = REFERENCE_METRICS | NO_REFERENCE_METRICS | DATASET_METRICS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute image restoration metrics for one image pair or paired folders."
    )
    parser.add_argument(
        "--pred",
        required=True,
        type=Path,
        help="Predicted/restored image file or folder.",
    )
    parser.add_argument(
        "--gt",
        type=Path,
        default=None,
        help="Ground-truth image file or folder. Required for PSNR/SSIM/LPIPS/FID.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["psnr", "ssim"],
        help=f"Metrics to compute. Supported: {', '.join(sorted(SUPPORTED_METRICS))}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV output path for per-image and average metrics.",
    )
    parser.add_argument(
        "--crop-border",
        type=int,
        default=0,
        help="Crop this many pixels from each border before computing paired metrics.",
    )
    parser.add_argument(
        "--y-channel",
        action="store_true",
        help="Compute PSNR/SSIM on luminance channel instead of RGB.",
    )
    parser.add_argument(
        "--resize-to-gt",
        action="store_true",
        help="Resize predicted image to GT size if shapes differ. Disabled by default.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for LPIPS/NIQE/BRISQUE, e.g. cpu or cuda.",
    )
    parser.add_argument(
        "--lpips-net",
        default="alex",
        choices=["alex", "vgg", "squeeze"],
        help="Backbone for LPIPS.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> List[str]:
    metrics = [metric.lower() for metric in args.metrics]
    unknown = sorted(set(metrics) - SUPPORTED_METRICS)
    if unknown:
        raise ValueError(f"Unsupported metric(s): {', '.join(unknown)}")

    if not args.pred.exists():
        raise FileNotFoundError(f"Prediction path does not exist: {args.pred}")

    needs_gt = any(metric in REFERENCE_METRICS or metric in DATASET_METRICS for metric in metrics)
    if needs_gt and args.gt is None:
        raise ValueError("--gt is required for PSNR, SSIM, LPIPS, and FID.")
    if args.gt is not None and not args.gt.exists():
        raise FileNotFoundError(f"Ground-truth path does not exist: {args.gt}")
    if args.crop_border < 0:
        raise ValueError("--crop-border must be >= 0.")

    return metrics


def collect_images(path: Path) -> Dict[str, Path]:
    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image file extension: {path}")
        return {path.name: path}

    images: Dict[str, Path] = {}
    for file_path in sorted(path.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
            rel = file_path.relative_to(path).as_posix()
            images[rel] = file_path
    if not images:
        raise ValueError(f"No image files found under: {path}")
    return images


def build_pairs(pred_path: Path, gt_path: Path) -> List[Tuple[str, Path, Path]]:
    pred_images = collect_images(pred_path)
    gt_images = collect_images(gt_path)

    common_keys = sorted(set(pred_images) & set(gt_images))
    if not common_keys:
        common_keys = match_by_filename(pred_images, gt_images)
    if not common_keys:
        raise ValueError(
            "No matching image names found between prediction and GT paths. "
            "Use the same relative filenames in both folders."
        )

    return [(key, pred_images[key], gt_images[key]) for key in common_keys]


def match_by_filename(pred_images: Dict[str, Path], gt_images: Dict[str, Path]) -> List[str]:
    gt_by_name = {path.name: key for key, path in gt_images.items()}
    matched: List[str] = []
    remapped_pred: Dict[str, Path] = {}

    for key, pred_file in pred_images.items():
        gt_key = gt_by_name.get(pred_file.name)
        if gt_key is None:
            continue
        matched.append(key)
        remapped_pred[key] = pred_file
        gt_images[key] = gt_images[gt_key]

    pred_images.update(remapped_pred)
    return sorted(matched)


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def resize_like(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    height, width = gt.shape[:2]
    pred_uint8 = np.clip(pred * 255.0, 0, 255).astype(np.uint8)
    resized = Image.fromarray(pred_uint8).resize((width, height), Image.BICUBIC)
    return np.asarray(resized, dtype=np.float32) / 255.0


def crop_border_pair(pred: np.ndarray, gt: np.ndarray, crop_border: int) -> Tuple[np.ndarray, np.ndarray]:
    if crop_border == 0:
        return pred, gt
    if min(pred.shape[0], pred.shape[1], gt.shape[0], gt.shape[1]) <= 2 * crop_border:
        raise ValueError(f"Crop border {crop_border} is too large for image shape {pred.shape}.")
    return (
        pred[crop_border:-crop_border, crop_border:-crop_border, ...],
        gt[crop_border:-crop_border, crop_border:-crop_border, ...],
    )


def to_y_channel(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]


def prepare_pair(
    pred_path: Path,
    gt_path: Path,
    crop_border: int,
    y_channel: bool,
    resize_to_gt: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    pred = load_rgb(pred_path)
    gt = load_rgb(gt_path)

    if pred.shape != gt.shape:
        if not resize_to_gt:
            raise ValueError(
                f"Shape mismatch for {pred_path.name}: pred {pred.shape}, gt {gt.shape}. "
                "Use --resize-to-gt if resizing is intended."
            )
        pred = resize_like(pred, gt)

    pred, gt = crop_border_pair(pred, gt, crop_border)
    if y_channel:
        pred = to_y_channel(pred)
        gt = to_y_channel(gt)
    return pred, gt


def compute_psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    mse = float(np.mean((pred - gt) ** 2))
    if mse == 0:
        return math.inf
    return 10.0 * math.log10(1.0 / mse)


def compute_ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity
    except ImportError as exc:
        raise ImportError("SSIM requires scikit-image. Install with: pip install scikit-image") from exc

    if pred.ndim == 2:
        return float(structural_similarity(gt, pred, data_range=1.0))
    return float(structural_similarity(gt, pred, channel_axis=-1, data_range=1.0))


class LpipsMetric:
    def __init__(self, device: str, net: str) -> None:
        try:
            import lpips
            import torch
        except ImportError as exc:
            raise ImportError(
                "LPIPS requires lpips and torch. Install with: pip install lpips torch torchvision"
            ) from exc

        self.torch = torch
        self.device = torch.device(device)
        self.model = lpips.LPIPS(net=net).to(self.device).eval()

    def __call__(self, pred_path: Path, gt_path: Path) -> float:
        pred = load_rgb(pred_path)
        gt = load_rgb(gt_path)
        if pred.shape != gt.shape:
            raise ValueError(
                f"LPIPS requires the same image shape: {pred_path.name}, pred {pred.shape}, gt {gt.shape}."
            )

        pred_tensor = self.to_tensor(pred)
        gt_tensor = self.to_tensor(gt)
        with self.torch.no_grad():
            value = self.model(pred_tensor, gt_tensor)
        return float(value.squeeze().detach().cpu().item())

    def to_tensor(self, image: np.ndarray):
        tensor = self.torch.from_numpy(image.transpose(2, 0, 1)).float()
        tensor = tensor.unsqueeze(0) * 2.0 - 1.0
        return tensor.to(self.device)


class PyiqaNoRefMetric:
    def __init__(self, metric_name: str, device: str) -> None:
        try:
            import pyiqa
            import torch
        except ImportError as exc:
            raise ImportError(
                f"{metric_name.upper()} requires pyiqa and torch. "
                "Install with: pip install pyiqa torch torchvision"
            ) from exc

        self.torch = torch
        self.metric = pyiqa.create_metric(metric_name, device=device)

    def __call__(self, image_path: Path) -> float:
        with self.torch.no_grad():
            value = self.metric(str(image_path))
        return float(value.squeeze().detach().cpu().item())


def compute_fid(pred_dir: Path, gt_dir: Path) -> float:
    if not pred_dir.is_dir() or not gt_dir.is_dir():
        raise ValueError("FID requires --pred and --gt to be folders.")
    try:
        from cleanfid import fid
    except ImportError as exc:
        raise ImportError("FID requires clean-fid. Install with: pip install clean-fid") from exc
    return float(fid.compute_fid(str(pred_dir), str(gt_dir)))


def mean(values: Iterable[float]) -> float:
    values = [value for value in values if not math.isnan(value)]
    if not values:
        return math.nan
    if any(math.isinf(value) for value in values):
        return math.inf
    return float(sum(values) / len(values))


def format_number(value: Optional[float]) -> str:
    if value is None:
        return ""
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf"
    return f"{value:.6f}"


def write_csv(output_path: Path, rows: Sequence[Dict[str, Optional[float]]], metrics: Sequence[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image", *metrics]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_number(row.get(key)) if key != "image" else row[key] for key in fieldnames})


def print_summary(rows: Sequence[Dict[str, Optional[float]]], metrics: Sequence[str]) -> None:
    print(f"Images: {len([row for row in rows if row.get('image') != 'AVERAGE'])}")
    print("Average:")
    for metric in metrics:
        avg_row = rows[-1] if rows and rows[-1].get("image") == "AVERAGE" else {}
        print(f"  {metric.upper()}: {format_number(avg_row.get(metric))}")


def append_average_row(rows: List[Dict[str, Optional[float]]], metrics: Sequence[str]) -> None:
    avg_row: Dict[str, Optional[float]] = {"image": "AVERAGE"}  # type: ignore[assignment]
    for metric in metrics:
        values = [row.get(metric) for row in rows if row.get(metric) is not None]
        avg_row[metric] = mean(value for value in values if value is not None)
    rows.append(avg_row)


def main() -> int:
    args = parse_args()
    metrics = validate_args(args)

    rows: List[Dict[str, Optional[float]]] = []

    paired_metrics = [metric for metric in metrics if metric in REFERENCE_METRICS]
    no_ref_metrics = [metric for metric in metrics if metric in NO_REFERENCE_METRICS]

    lpips_metric = LpipsMetric(args.device, args.lpips_net) if "lpips" in paired_metrics else None
    niqe_metric = PyiqaNoRefMetric("niqe", args.device) if "niqe" in no_ref_metrics else None
    brisque_metric = PyiqaNoRefMetric("brisque", args.device) if "brisque" in no_ref_metrics else None

    if paired_metrics:
        pairs = build_pairs(args.pred, args.gt)
        for image_key, pred_path, gt_path in pairs:
            row: Dict[str, Optional[float]] = {"image": image_key}  # type: ignore[assignment]

            pred_array: Optional[np.ndarray] = None
            gt_array: Optional[np.ndarray] = None
            if "psnr" in paired_metrics or "ssim" in paired_metrics:
                pred_array, gt_array = prepare_pair(
                    pred_path,
                    gt_path,
                    args.crop_border,
                    args.y_channel,
                    args.resize_to_gt,
                )
            if "psnr" in paired_metrics and pred_array is not None and gt_array is not None:
                row["psnr"] = compute_psnr(pred_array, gt_array)
            if "ssim" in paired_metrics and pred_array is not None and gt_array is not None:
                row["ssim"] = compute_ssim(pred_array, gt_array)
            if lpips_metric is not None:
                row["lpips"] = lpips_metric(pred_path, gt_path)

            rows.append(row)

    if no_ref_metrics:
        pred_images = collect_images(args.pred)
        existing_by_image = {str(row["image"]): row for row in rows}
        for image_key, pred_path in pred_images.items():
            row = existing_by_image.get(image_key)
            if row is None:
                row = {"image": image_key}  # type: ignore[assignment]
                rows.append(row)
            if niqe_metric is not None:
                row["niqe"] = niqe_metric(pred_path)
            if brisque_metric is not None:
                row["brisque"] = brisque_metric(pred_path)

    if "fid" in metrics:
        fid_value = compute_fid(args.pred, args.gt)
        if rows:
            for row in rows:
                row.setdefault("fid", None)
        rows.append({"image": "FID", "fid": fid_value})  # type: ignore[list-item]

    if not rows:
        raise RuntimeError("No metrics were computed.")

    average_metrics = [metric for metric in metrics if metric != "fid"]
    if average_metrics:
        append_average_row(rows, average_metrics)

    print_summary(rows, average_metrics if average_metrics else ["fid"])
    if "fid" in metrics:
        fid_rows = [row for row in rows if row.get("image") == "FID"]
        if fid_rows:
            print(f"  FID: {format_number(fid_rows[-1].get('fid'))}")

    if args.output is not None:
        write_csv(args.output, rows, metrics)
        print(f"Saved CSV: {args.output}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
