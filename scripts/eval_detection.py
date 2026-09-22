"""PSNR, SSIM, and mAP@50 for a longitudinal SR checkpoint.

Reconstruction is compared with the high-resolution depth image. When a frozen
detector checkpoint is supplied, the same detector is run on the reconstruction
and scored against YOLO labels at IoU 0.50. Those labels use the original HR
coordinates because LP-SRNet restores the longitudinal axis.
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basicsr.archs.lpsrnet_arch import LPSRNet
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim
from basicsr.utils.longitudinal_infer import longitudinal_tile_forward

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


def _read_gray(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f'Failed to read {path}.')
    return image


def _load_network(weights, scale, device):
    checkpoint = torch.load(weights, map_location='cpu')
    if isinstance(checkpoint, dict):
        state = checkpoint.get('params_ema', checkpoint.get('params', checkpoint))
    else:
        state = checkpoint
    net = LPSRNet(upscaling_factor=scale)
    net.load_state_dict(state, strict=True)
    net.eval().to(device)
    return net


def _yolo_to_xyxy(path, height, width):
    boxes = []
    if not os.path.isfile(path):
        return boxes
    with open(path, 'r', encoding='utf-8') as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls, cx, cy, bw, bh = [float(part) for part in parts[:5]]
            x1 = (cx - bw / 2.0) * width
            y1 = (cy - bh / 2.0) * height
            x2 = (cx + bw / 2.0) * width
            y2 = (cy + bh / 2.0) * height
            boxes.append((int(cls), np.array([x1, y1, x2, y2], dtype=np.float32)))
    return boxes


def _iou(box, boxes):
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.float32)
    others = np.stack(boxes, axis=0)
    x1 = np.maximum(box[0], others[:, 0])
    y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2])
    y2 = np.minimum(box[3], others[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    area_b = np.maximum(0, others[:, 2] - others[:, 0]) * np.maximum(0, others[:, 3] - others[:, 1])
    return inter / np.maximum(area_a + area_b - inter, 1e-6)


def _average_precision(recall, precision):
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    for index in range(precision.size - 1, 0, -1):
        precision[index - 1] = max(precision[index - 1], precision[index])
    change = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[change + 1] - recall[change]) * precision[change + 1]))


def mean_ap50(predictions, ground_truth, num_classes):
    """predictions[image] = list of (score, cls, xyxy); ground_truth likewise without score."""
    aps = []
    for cls in range(num_classes):
        scores = []
        matched = []
        n_gt = 0
        for image_id, gt_boxes in enumerate(ground_truth):
            gt_cls = [box for label, box in gt_boxes if label == cls]
            n_gt += len(gt_cls)
            used = np.zeros(len(gt_cls), dtype=bool)
            candidates = [(score, box) for score, label, box in predictions[image_id] if label == cls]
            candidates.sort(key=lambda item: item[0], reverse=True)
            for score, box in candidates:
                scores.append(score)
                ious = _iou(box, gt_cls)
                order = np.argsort(-ious) if ious.size else []
                hit = False
                for gt_index in order:
                    if ious[gt_index] < 0.5:
                        break
                    if not used[gt_index]:
                        used[gt_index] = True
                        hit = True
                        break
                matched.append(hit)
        if n_gt == 0:
            continue
        if not scores:
            aps.append(0.0)
            continue
        order = np.argsort(-np.asarray(scores))
        matched = np.asarray(matched, dtype=np.float32)[order]
        tp = np.cumsum(matched)
        fp = np.cumsum(1.0 - matched)
        aps.append(_average_precision(tp / n_gt, tp / np.maximum(tp + fp, 1e-6)))
    if not aps:
        return float('nan')
    return float(np.mean(aps))


def main():
    parser = argparse.ArgumentParser(description='Evaluate longitudinal SR and optional detection.')
    parser.add_argument('--weights', required=True)
    parser.add_argument('--lq-dir', required=True)
    parser.add_argument('--gt-dir', required=True)
    parser.add_argument('--label-dir', default=None)
    parser.add_argument('--scale', type=int, choices=(4, 8), required=True)
    parser.add_argument('--detector', default=None, help='Frozen LPD-Net / Ultralytics checkpoint.')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--num-classes', type=int, default=5)
    parser.add_argument('--tile-h', type=int, default=128)
    parser.add_argument('--tile-w', type=int, default=512)
    parser.add_argument('--tile-overlap', type=int, default=16)
    parser.add_argument('--save-dir', default=None)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    names = sorted(name for name in os.listdir(args.lq_dir) if os.path.splitext(name)[1].lower() in IMAGE_EXTS)
    net = _load_network(args.weights, args.scale, args.device)
    detector = None
    if args.detector:
        from ultralytics import YOLO
        detector = YOLO(args.detector)

    psnr_sum = 0.0
    ssim_sum = 0.0
    predictions = []
    ground_truth = []
    for name in tqdm(names):
        low_res = _read_gray(os.path.join(args.lq_dir, name))
        high_res = _read_gray(os.path.join(args.gt_dir, name))
        high_res = high_res[:low_res.shape[0] * args.scale, :low_res.shape[1]]
        tensor = torch.from_numpy(low_res).float().div(255.0)[None, None].to(args.device)
        with torch.no_grad():
            output = longitudinal_tile_forward(net, tensor, args.scale, (args.tile_h, args.tile_w), args.tile_overlap)
        restored = output.squeeze().clamp(0, 1).mul(255.0).round().to(torch.uint8).cpu().numpy()
        psnr_sum += calculate_psnr(restored, high_res, crop_border=args.scale, test_y_channel=False)
        ssim_sum += calculate_ssim(restored, high_res, crop_border=args.scale, test_y_channel=False)
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            cv2.imwrite(os.path.join(args.save_dir, name), restored)
        if detector is not None:
            color = cv2.cvtColor(restored, cv2.COLOR_GRAY2BGR)
            result = detector.predict(color, imgsz=args.imgsz, conf=0.001, verbose=False)[0]
            image_preds = []
            if result.boxes is not None and len(result.boxes):
                xyxy = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                image_preds = [(float(score), int(cls), xyxy[index]) for index, (score, cls) in enumerate(zip(scores, classes))]
            predictions.append(image_preds)
            stem = os.path.splitext(name)[0]
            ground_truth.append(_yolo_to_xyxy(os.path.join(args.label_dir, f'{stem}.txt'), restored.shape[0], restored.shape[1]))

    count = max(len(names), 1)
    print(f'PSNR {psnr_sum / count:.4f} dB  SSIM {ssim_sum / count:.4f}  images {len(names)}')
    if detector is not None:
        print(f'mAP@50 {mean_ap50(predictions, ground_truth, args.num_classes):.4f}')


if __name__ == '__main__':
    main()
