# preprocess.py
# Smart preprocessing - only processes images that actually need it.
#
# What we learned from first attempt:
#   Preprocessing all images made things worse.
#   Good quality images do not need preprocessing.
#   Only bad quality images should be preprocessed.
#
# Quality thresholds:
#   contrast  < 30  → image is too faded
#   sharpness < 100 → image is too blurry
#   Both must be bad to trigger preprocessing.

import cv2
import os
import shutil


def check_quality(image_path):
    img       = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    contrast  = round(float(img.std()), 2)
    sharpness = round(cv2.Laplacian(img, cv2.CV_64F).var(), 2)
    return contrast, sharpness


def preprocess(image_path, save_path):
    img  = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    denoised = cv2.fastNlMeansDenoising(enhanced, h=10)

    _, binary = cv2.threshold(
        denoised, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    cv2.imwrite(save_path, binary)


if __name__ == "__main__":
    from config import DATA, RESULTS

    img_dir       = os.path.join(DATA, "degraded", "images")
    processed_dir = os.path.join(RESULTS, "degraded_processed")
    os.makedirs(processed_dir, exist_ok=True)

    print("Smart Preprocessing — Degraded Images")
    print("=" * 55)

    for img_name in sorted(os.listdir(img_dir)):
        if not img_name.endswith('.png'):
            continue

        original_path  = os.path.join(img_dir, img_name)
        processed_path = os.path.join(processed_dir, img_name)

        contrast, sharpness = check_quality(original_path)

        print(f"\n  Image     : {img_name}")
        print(f"  Contrast  : {contrast}")
        print(f"  Sharpness : {sharpness}")

        if contrast < 30 and sharpness < 100:
            preprocess(original_path, processed_path)
            print(f"  Decision  : PREPROCESSED - image quality was bad")
        else:
            shutil.copy(original_path, processed_path)
            print(f"  Decision  : SKIPPED - image quality is good enough")

    print(f"\n  Done. Images saved to: {processed_dir}")