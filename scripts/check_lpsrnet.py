"""Shape, gradient, and degradation checks for the 4x and 8x longitudinal models."""

import importlib.util
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basicsr.archs.lpsrnet_arch import LPSRNet
from basicsr.data.longitudinal_dataset import _clip_boxes_to_crop
from basicsr.models.frozen_detector import detection_objective
from basicsr.utils.longitudinal_infer import longitudinal_tile_forward

_spec = importlib.util.spec_from_file_location(
    'degrade_longitudinal', os.path.join(os.path.dirname(__file__), 'degrade_longitudinal.py'))
_degrade = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_degrade)
degrade_image = _degrade.degrade_image
gaussian_kernel = _degrade.gaussian_kernel


def count_parameters(net):
    return sum(parameter.numel() for parameter in net.parameters())


def check_network(scale):
    net = LPSRNet(upscaling_factor=scale, use_ea=True, use_pdcfn=True, use_pps=True)
    net.train()
    low_res = torch.rand(2, 1, 16, 24, requires_grad=True)
    output = net(low_res)
    expected = (2, 1, 16 * scale, 24)
    if tuple(output.shape) != expected:
        raise AssertionError(f'x{scale} output {tuple(output.shape)} != {expected}')
    output.mean().backward()
    if low_res.grad is None or not torch.isfinite(low_res.grad).all():
        raise AssertionError(f'x{scale} backward did not produce finite input gradients.')
    baseline = LPSRNet(upscaling_factor=scale, use_ea=False, use_pdcfn=False, use_pps=False)
    print(f'x{scale}: LP-SRNet {count_parameters(net) / 1000:.2f} K, '
          f'longitudinal SMFANet {count_parameters(baseline) / 1000:.2f} K, output {tuple(output.shape)}')


def check_degrade():
    length, sigma = gaussian_kernel(4)[1:]
    if (length, sigma) != (11, 1.5):
        raise AssertionError((length, sigma))
    length, sigma = gaussian_kernel(8)[1:]
    if (length, sigma) != (23, 3.5):
        raise AssertionError((length, sigma))
    image = np.tile(np.linspace(0, 255, 5000, dtype=np.uint8)[:, None], (1, 64))
    low4 = degrade_image(image, 4)
    low8 = degrade_image(image, 8)
    if low4.shape != (1250, 64) or low8.shape != (625, 64):
        raise AssertionError((low4.shape, low8.shape))
    print(f'degrade: x4 {low4.shape}, x8 {low8.shape}')


def check_boxes_and_tiles():
    boxes = np.array([[0, 0.25, 0.50, 0.10, 0.10]], dtype=np.float32)
    cropped = _clip_boxes_to_crop(boxes, 40, 10, 20, 30, 100, 100)
    if cropped.shape != (1, 5):
        raise AssertionError(cropped)
    net = LPSRNet(upscaling_factor=4).eval()
    low_res = torch.rand(1, 1, 20, 30)
    with torch.no_grad():
        tiled = longitudinal_tile_forward(net, low_res, 4, (8, 16), overlap=2)
        full = net(low_res)
    if tiled.shape != full.shape:
        raise AssertionError((tiled.shape, full.shape))
    objective = detection_objective(torch.tensor(1.0), torch.tensor(1.0), torch.tensor(1.0))
    if not torch.isclose(objective, torch.tensor(9.5)):
        raise AssertionError(objective)
    print('boxes, tiled forward, and detection objective passed')


if __name__ == '__main__':
    check_degrade()
    check_network(4)
    check_network(8)
    check_boxes_and_tiles()
    print('LP-SRNet checks passed.')
