# upscale.py
# Upscales small scene text images before running OCR.
#
# Why we need this:
#   scene_003, scene_004, scene_005 scored 0% accuracy.
#   Images are only 109x55 pixels.
#   PaddleOCR needs minimum resolution to detect text.
#   We make the image 3x bigger before running OCR.

import cv2
import os
import shutil


def check_size(image_path):
    img    = cv2.imread(image_path)
    h, w   = img.shape[:2]
    return w, h


def upscale(image_path, save_path, scale=3):
    img      = cv2.imread(image_path)
    h, w     = img.shape[:2]
    upscaled = cv2.resize(
        img,
        (w * scale, h * scale),
        interpolation=cv2.INTER_CUBIC
    )
    cv2.imwrite(save_path, upscaled)


if __name__ == "__main__":
    from config import DATA, RESULTS

    img_dir      = os.path.join(DATA, "scene_text", "images")
    upscaled_dir = os.path.join(RESULTS, "scene_upscaled")
    os.makedirs(upscaled_dir, exist_ok=True)

    print("Upscaling Scene Text Images")
    print("=" * 55)

    for img_name in sorted(os.listdir(img_dir)):
        if not img_name.endswith('.png'):
            continue

        original_path = os.path.join(img_dir, img_name)
        upscaled_path = os.path.join(upscaled_dir, img_name)

        w, h = check_size(original_path)

        print(f"\n  Image    : {img_name}")
        print(f"  Size     : {w}x{h} pixels")

        if w < 200 or h < 200:
            upscale(original_path, upscaled_path)
            print(f"  Decision : UPSCALED to {w*3}x{h*3}")
        else:
            shutil.copy(original_path, upscaled_path)
            print(f"  Decision : SKIPPED - already large enough")

    print(f"\n  Done. Images saved to: {upscaled_dir}")