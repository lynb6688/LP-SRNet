"""Create longitudinally blurred and decimated depth images.

The degradation follows the article. A normalized one-dimensional Gaussian is
applied only along the travel direction, then every s-th profile is kept.
Scale 4 uses an 11-tap kernel with sigma 1.5. Scale 8 uses a 23-tap kernel
with sigma 3.5. Reflection padding is used, and the floating-point result is
rounded, clipped, and stored as an 8-bit image. Width is unchanged.

Images must be arranged as height = longitudinal and width = transverse, so an
HR frame of 4,096 x 5,000 (transverse x longitudinal) has array shape
(5000, 4096). If a file stores travel along its width, pass
``--longitudinal-axis width``; the saved image is transposed to the height
convention expected by training.
"""

import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm

SCALES = {
    4: (11, 1.5),
    8: (23, 3.5),
}
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


def gaussian_kernel(scale):
    """Return a column kernel and the (length, sigma) pair for this scale."""
    if int(scale) not in SCALES:
        raise ValueError(f'Only scales 4 and 8 are defined, got {scale}.')
    length, sigma = SCALES[int(scale)]
    coords = np.arange(length, dtype=np.float64) - (length - 1) / 2.0
    kernel = np.exp(-0.5 * (coords / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel.astype(np.float32).reshape(-1, 1), length, sigma


def degrade_image(image, scale):
    """Blur and decimate a grayscale image along axis 0.

    Args:
        image (ndarray): uint8 array of shape (H, W) or (H, W, 1).
        scale (int): 4 or 8.

    Returns:
        ndarray: uint8 low-resolution image of shape (H // scale, W).
    """
    if image.ndim == 3:
        image = image[..., 0]
    kernel, _, _ = gaussian_kernel(scale)
    if image.shape[0] < int(scale):
        raise ValueError(f'Image height {image.shape[0]} is shorter than the longitudinal scale {scale}.')
    usable = image.shape[0] - (image.shape[0] % int(scale))
    image = image[:usable]
    blurred = cv2.filter2D(image.astype(np.float32), -1, kernel, borderType=cv2.BORDER_REFLECT_101)
    low_res = blurred[::int(scale)]
    return np.clip(np.rint(low_res), 0, 255).astype(np.uint8)


def _iter_images(root, recursive):
    if recursive:
        for dirpath, _, filenames in os.walk(root):
            for name in sorted(filenames):
                if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                    yield os.path.join(dirpath, name)
    else:
        for name in sorted(os.listdir(root)):
            path = os.path.join(root, name)
            if os.path.isfile(path) and os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                yield path


def main():
    parser = argparse.ArgumentParser(description='Longitudinal Gaussian degradation for depth images.')
    parser.add_argument('--hr-dir', required=True, help='Directory of high-resolution depth images.')
    parser.add_argument('--out-dir', required=True, help='Directory for the low-resolution images.')
    parser.add_argument('--scale', type=int, choices=(4, 8), required=True)
    parser.add_argument('--longitudinal-axis', choices=('height', 'width'), default='height')
    parser.add_argument('--hr-out', default=None, help='If set, also write HR images in height-longitudinal layout.')
    parser.add_argument('--recursive', action='store_true')
    args = parser.parse_args()

    paths = list(_iter_images(args.hr_dir, args.recursive))
    if not paths:
        raise SystemExit(f'No images found in {args.hr_dir}.')
    for path in tqdm(paths, desc=f'x{args.scale}'):
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f'Failed to read {path}.')
        if args.longitudinal_axis == 'width':
            image = np.ascontiguousarray(image.T)
        low_res = degrade_image(image, args.scale)
        relative = os.path.relpath(path, args.hr_dir)
        destination = os.path.join(args.out_dir, relative)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        if not cv2.imwrite(destination, low_res):
            raise RuntimeError(f'Failed to write {destination}.')
        if args.hr_out:
            hr_destination = os.path.join(args.hr_out, relative)
            os.makedirs(os.path.dirname(hr_destination), exist_ok=True)
            if not cv2.imwrite(hr_destination, image):
                raise RuntimeError(f'Failed to write {hr_destination}.')


if __name__ == '__main__':
    main()
