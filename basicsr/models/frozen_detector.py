"""Frozen distress detector used by task-driven fine-tuning.

The published detection objective is

    L_det = 7.5 L_cls + 0.5 L_box + 1.5 L_dfl.

LPD-Net is loaded from an Ultralytics checkpoint. Its classification, box, and
distribution-focal gains are overwritten with those coefficients. Detector
parameters stay frozen, while gradients still flow back to the reconstructed image.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def detection_objective(cls_loss, box_loss, dfl_loss, cls_weight=7.5, box_weight=0.5, dfl_weight=1.5):
    """Combine the three detection terms with the article's coefficients."""
    return cls_weight * cls_loss + box_weight * box_loss + dfl_weight * dfl_loss


class FrozenDetector(nn.Module):
    """Ultralytics LPD-Net (or another YOLO-style detector) with frozen weights."""

    def __init__(self, weights, cls_weight=7.5, box_weight=0.5, dfl_weight=1.5, imgsz=640):
        super().__init__()
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                'Task-driven fine-tuning needs ultralytics to load the frozen LPD-Net checkpoint. '
                'Install it with: pip install ultralytics') from exc

        detector = YOLO(weights).model
        for parameter in detector.parameters():
            parameter.requires_grad = False
        self.detector = detector
        self.cls_weight = float(cls_weight)
        self.box_weight = float(box_weight)
        self.dfl_weight = float(dfl_weight)
        self.imgsz = None if imgsz in (None, 0, 'none') else int(imgsz)
        self.in_channels = _first_conv_in_channels(detector)

    def train(self, mode=True):
        super().train(mode)
        # Keep the detector graph in training mode so it returns raw predictions,
        # but freeze normalization statistics.
        self.detector.train()
        for module in self.detector.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def forward(self, images):
        raise RuntimeError('Use detection_loss(); the frozen detector is not an image generator.')

    def detection_loss(self, images, labels, bboxes, n_targets):
        """Return the scalar detection objective for a reconstructed batch.

        Args:
            images (Tensor): (B, C, H, W) reconstructed depth in the GT crop.
            labels (Tensor): (B, M) class ids.
            bboxes (Tensor): (B, M, 4) YOLO xywh normalized to the crop.
            n_targets (Tensor): (B,) valid box counts.
        """
        self._apply_gains()
        batch_idx, cls, boxes = _flatten_targets(labels, bboxes, n_targets)
        if cls.numel() == 0:
            return images.sum() * 0

        adapted = _match_channels(images, self.in_channels)
        if self.imgsz is not None:
            adapted, boxes = _letterbox(adapted, boxes, self.imgsz)
        yolo_batch = {
            'img': adapted,
            'cls': cls.float().unsqueeze(1),
            'bboxes': boxes,
            'batch_idx': batch_idx.float(),
        }
        loss, _items = self.detector.loss(yolo_batch)
        return loss / adapted.shape[0]

    def _apply_gains(self):
        args = self.detector.args
        args.box = self.box_weight
        args.cls = self.cls_weight
        args.dfl = self.dfl_weight
        criterion = getattr(self.detector, 'criterion', None)
        if criterion is not None and hasattr(criterion, 'hyp'):
            criterion.hyp.box = self.box_weight
            criterion.hyp.cls = self.cls_weight
            criterion.hyp.dfl = self.dfl_weight


def _first_conv_in_channels(module):
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            return child.in_channels
    return 3


def _match_channels(images, in_channels):
    channels = images.shape[1]
    if channels == in_channels:
        return images
    if channels == 1 and in_channels == 3:
        return images.repeat(1, 3, 1, 1)
    if channels == 3 and in_channels == 1:
        return images.mean(dim=1, keepdim=True)
    raise RuntimeError(f'Cannot adapt a {channels}-channel reconstruction to a {in_channels}-channel detector.')


def _flatten_targets(labels, bboxes, n_targets):
    batch_idx = []
    cls = []
    boxes = []
    for index in range(labels.shape[0]):
        count = int(n_targets[index])
        if count <= 0:
            continue
        batch_idx.append(labels.new_full((count,), index))
        cls.append(labels[index, :count])
        boxes.append(bboxes[index, :count])
    if not cls:
        empty = labels.new_zeros((0,))
        return empty, empty, bboxes.new_zeros((0, 4))
    return torch.cat(batch_idx, 0), torch.cat(cls, 0), torch.cat(boxes, 0)


def _letterbox(images, boxes, size):
    """Differentiable letterbox. Boxes are xywh normalized to the input image."""
    _, _, height, width = images.shape
    ratio = size / max(height, width)
    resized_h = max(int(round(height * ratio)), 1)
    resized_w = max(int(round(width * ratio)), 1)
    images = F.interpolate(images, size=(resized_h, resized_w), mode='bilinear', align_corners=False)
    pad_h = size - resized_h
    pad_w = size - resized_w
    top = pad_h // 2
    left = pad_w // 2
    images = F.pad(images, (left, pad_w - left, top, pad_h - top))

    if boxes.numel():
        xywh = boxes.clone()
        xywh[:, 0] = boxes[:, 0] * width * ratio + left
        xywh[:, 1] = boxes[:, 1] * height * ratio + top
        xywh[:, 2] = boxes[:, 2] * width * ratio
        xywh[:, 3] = boxes[:, 3] * height * ratio
        xywh = xywh / float(size)
        boxes = xywh.clamp(0, 1)
    return images, boxes
