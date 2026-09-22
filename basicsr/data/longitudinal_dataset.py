"""Paired depth-image dataset for longitudinal super-resolution.

Low-resolution images are already degraded along height. Height is the
direction of travel and width is the transverse profile, so a scale of 4 or 8
relates GT height to LQ height and leaves width unchanged. Rotation is not
applied, because it would swap the two axes.
"""

import os
import random

import numpy as np
import torch
from torch.utils import data as data

from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY

# JTG 5210-2018 categories used by ZJNUCrack, in label-id order.
DISTRESS_CLASSES = (
    'longitudinal_cracking',  # C00
    'transverse_cracking',  # C01
    'map_cracking',  # C10
    'pothole',  # C11
    'sealed_cracking',  # C22
)


def _as_hwc(img):
    if img.ndim == 2:
        img = img[..., None]
    return img


def _read_yolo_labels(path):
    if path is None or not os.path.isfile(path):
        return np.zeros((0, 5), dtype=np.float32)
    rows = []
    with open(path, 'r', encoding='utf-8') as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            rows.append([float(part) for part in parts[:5]])
    if not rows:
        return np.zeros((0, 5), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def _clip_boxes_to_crop(boxes, top, left, crop_h, crop_w, full_h, full_w):
    """Convert full-image YOLO boxes to crop-normalized xywh, dropping misses."""
    if boxes.shape[0] == 0:
        return boxes
    cx = boxes[:, 1] * full_w
    cy = boxes[:, 2] * full_h
    bw = boxes[:, 3] * full_w
    bh = boxes[:, 4] * full_h
    x1 = np.clip(cx - bw / 2.0, left, left + crop_w)
    y1 = np.clip(cy - bh / 2.0, top, top + crop_h)
    x2 = np.clip(cx + bw / 2.0, left, left + crop_w)
    y2 = np.clip(cy + bh / 2.0, top, top + crop_h)
    keep = ((x2 - x1) > 1.0) & ((y2 - y1) > 1.0)
    if not np.any(keep):
        return np.zeros((0, 5), dtype=np.float32)
    x1 = (x1[keep] - left) / crop_w
    x2 = (x2[keep] - left) / crop_w
    y1 = (y1[keep] - top) / crop_h
    y2 = (y2[keep] - top) / crop_h
    out = np.stack((
        boxes[keep, 0],
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
        x2 - x1,
        y2 - y1,
    ), axis=1)
    return out.astype(np.float32)


def _longitudinal_crop(img_gt, img_lq, gt_h, gt_w, scale):
    h_lq, w_lq = img_lq.shape[:2]
    h_gt, w_gt = img_gt.shape[:2]
    if h_gt < h_lq * scale or w_gt < w_lq:
        raise ValueError(f'GT {(h_gt, w_gt)} is smaller than the longitudinal pair of LQ {(h_lq, w_lq)} at scale {scale}.')
    img_gt = img_gt[:h_lq * scale, :w_lq]
    h_gt, w_gt = img_gt.shape[:2]
    lq_h = gt_h // scale
    lq_w = gt_w
    if h_lq < lq_h or w_lq < lq_w:
        raise ValueError(
            f'LQ {(h_lq, w_lq)} cannot supply a {lq_h}x{lq_w} patch. Reduce gt_size_h/gt_size_w.')
    top = random.randint(0, h_lq - lq_h)
    left = random.randint(0, w_lq - lq_w)
    img_lq = img_lq[top:top + lq_h, left:left + lq_w]
    gt_top = top * scale
    img_gt = img_gt[gt_top:gt_top + gt_h, left:left + gt_w]
    return img_gt, img_lq, gt_top, left


@DATASET_REGISTRY.register()
class LongitudinalPairedDataset(data.Dataset):
    """LQ/GT depth pairs, with optional YOLO labels on the GT image.

    Args:
        opt (dict): Dataset options. Besides the usual BasicSR keys, this
            accepts ``gt_size_h``, ``gt_size_w``, ``use_vflip``, ``dataroot_label``,
            and ``max_det``. ``gt_size`` is used as both sides when the explicit
            sizes are omitted. Labels are class, cx, cy, w, h, normalized to the
            full GT image.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.gt_folder = opt['dataroot_gt']
        self.lq_folder = opt['dataroot_lq']
        self.label_folder = opt.get('dataroot_label', None)
        self.filename_tmpl = opt.get('filename_tmpl', '{}')
        self.max_det = int(opt.get('max_det', 64))
        gt_size = opt.get('gt_size', 256)
        self.gt_h = int(opt.get('gt_size_h', gt_size if not isinstance(gt_size, (list, tuple)) else gt_size[0]))
        self.gt_w = int(opt.get('gt_size_w', gt_size if not isinstance(gt_size, (list, tuple)) else gt_size[1]))

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif opt.get('meta_info_file', None) is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'], opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = int(self.opt['scale'])
        gt_path = self.paths[index]['gt_path']
        lq_path = self.paths[index]['lq_path']
        img_gt = _as_hwc(imfrombytes(self.file_client.get(gt_path, 'gt'), flag='grayscale', float32=True))
        img_lq = _as_hwc(imfrombytes(self.file_client.get(lq_path, 'lq'), flag='grayscale', float32=True))

        boxes = None
        if self.label_folder:
            stem = os.path.splitext(os.path.basename(gt_path))[0]
            boxes = _read_yolo_labels(os.path.join(self.label_folder, f'{stem}.txt'))

        if self.opt['phase'] == 'train':
            full_h, full_w = img_gt.shape[:2]
            img_gt, img_lq, top, left = _longitudinal_crop(img_gt, img_lq, self.gt_h, self.gt_w, scale)
            if boxes is not None:
                boxes = _clip_boxes_to_crop(boxes, top, left, self.gt_h, self.gt_w, full_h, full_w)
            if self.opt.get('use_hflip', True) and random.random() < 0.5:
                img_gt = np.flip(img_gt, axis=1).copy()
                img_lq = np.flip(img_lq, axis=1).copy()
                if boxes is not None and boxes.shape[0] > 0:
                    boxes[:, 1] = 1.0 - boxes[:, 1]
            if self.opt.get('use_vflip', True) and random.random() < 0.5:
                img_gt = np.flip(img_gt, axis=0).copy()
                img_lq = np.flip(img_lq, axis=0).copy()
                if boxes is not None and boxes.shape[0] > 0:
                    boxes[:, 2] = 1.0 - boxes[:, 2]
        else:
            img_gt = img_gt[:img_lq.shape[0] * scale, :img_lq.shape[1]]
            if boxes is not None:
                full_h, full_w = img_gt.shape[:2]
                boxes = _clip_boxes_to_crop(boxes, 0, 0, full_h, full_w, full_h, full_w)

        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=False, float32=True)
        sample = {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}
        if boxes is not None:
            sample.update(_pad_targets(boxes, self.max_det))
        return sample

    def __len__(self):
        return len(self.paths)


def _pad_targets(boxes, max_det):
    if boxes.shape[0] > max_det:
        area = boxes[:, 3] * boxes[:, 4]
        boxes = boxes[np.argsort(area)[-max_det:]]
    n = boxes.shape[0]
    bboxes = torch.zeros(max_det, 4, dtype=torch.float32)
    labels = torch.zeros(max_det, dtype=torch.long)
    if n:
        bboxes[:n] = torch.from_numpy(boxes[:, 1:5])
        labels[:n] = torch.from_numpy(boxes[:, 0].astype(np.int64))
    return {'bboxes': bboxes, 'labels': labels, 'n_targets': torch.tensor(n, dtype=torch.long)}
