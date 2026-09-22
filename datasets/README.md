# Depth-image layout

SACrack is used for reconstruction pretraining. ZJNUCrack is used for task-driven fine-tuning and testing. Split each acquisition segment before degradation, about 7:2:1, and keep every scale of an image in the same split.

## Download

The depth images are shared on Baidu Netdisk. The shared file is **Def**.

链接: https://pan.baidu.com/s/1NNIJG4tMBOV0SpUeH5Vd-A?pwd=abcd  
提取码: `abcd`

```
datasets/
  SACrack/
    train/HR/  train/LR/x4/  train/LR/x8/
    val/HR/    val/LR/x4/    val/LR/x8/
    test/HR/   test/LR/x4/   test/LR/x8/
  ZJNUCrack/
    train/HR/  train/LR/x4/  train/LR/x8/  train/labels/
    val/HR/    val/LR/x4/    val/LR/x8/
    test/HR/   test/LR/x4/   test/LR/x8/   test/labels/
```

HR and LR files share the same file name. A label file has the same stem and uses YOLO lines `class cx cy w h`, normalized to the HR image. Class ids are 0 C00 longitudinal cracking, 1 C01 transverse cracking, 2 C10 map cracking, 3 C11 pothole, and 4 C22 sealed cracking.

Store images with height along the direction of travel and width across the profile. A 4,096 x 5,000 transverse-by-longitudinal frame is an array of shape (5000, 4096).

```
python scripts/degrade_longitudinal.py --hr-dir datasets/SACrack/train/HR --out-dir datasets/SACrack/train/LR/x4 --scale 4
python scripts/degrade_longitudinal.py --hr-dir datasets/SACrack/train/HR --out-dir datasets/SACrack/train/LR/x8 --scale 8
```

Repeat those commands for every split. If travel is stored along image width, add `--longitudinal-axis width`. The script then saves both the LR image and, with `--hr-out`, the HR image in the height-longitudinal layout used by training.
