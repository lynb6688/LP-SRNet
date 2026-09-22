"""Tiled inference for images whose longitudinal axis is too tall for one pass."""

import torch


def _positions(length, tile, overlap):
    if tile is None or tile <= 0 or length <= tile:
        return [0]
    stride = tile - overlap
    if stride <= 0:
        raise ValueError(f'Tile size {tile} must be larger than overlap {overlap}.')
    positions = list(range(0, length - tile + 1, stride))
    last = length - tile
    if positions[-1] != last:
        positions.append(last)
    return positions


def longitudinal_tile_forward(net, lq, scale, tile_size, overlap=16):
    """Run a fully convolutional longitudinal SR network on LQ tiles.

    Args:
        net (nn.Module): Network mapping (N, C, h, w) to (N, C, h * scale, w).
        lq (Tensor): Low-resolution depth image.
        scale (int): Longitudinal scale.
        tile_size (int | list[int]): Tile height and width on the LQ image.
            A single int is used for both sides.
        overlap (int): LQ-pixel overlap between tiles. Default: 16.
    """
    if tile_size is None:
        return net(lq)
    if isinstance(tile_size, int):
        tile_h = tile_w = tile_size
    else:
        tile_h, tile_w = int(tile_size[0]), int(tile_size[1])
    overlap = int(overlap)
    _, _, height, width = lq.shape
    vertical = _positions(height, tile_h, overlap)
    horizontal = _positions(width, tile_w, overlap)

    output = None
    weight = None
    for top in vertical:
        h_end = height if height <= tile_h else top + tile_h
        for left in horizontal:
            w_end = width if width <= tile_w else left + tile_w
            pred = net(lq[:, :, top:h_end, left:w_end])
            if output is None:
                output = lq.new_zeros(lq.shape[0], pred.shape[1], height * scale, width)
                weight = lq.new_zeros(1, 1, height * scale, width)
            out_top = top * scale
            out_h = pred.shape[2]
            output[:, :, out_top:out_top + out_h, left:w_end] += pred
            weight[:, :, out_top:out_top + out_h, left:w_end] += 1
    return output / weight.clamp_min(1)
