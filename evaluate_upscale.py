# evaluate_upscaled.py
# Runs OCR on upscaled scene text images
# and compares with original results.

import os
import csv

from config     import DATA, RESULTS
from ocr_engine import load_model, extract_text
from metrics    import (
    word_accuracy,
    calculate_cer,
    calculate_wer,
    clean
)


def read_ground_truth(gt_dir, image_name):
    index   = image_name.split('_')[-1].replace('.png', '')
    matches = [
        f for f in os.listdir(gt_dir)
        if f.endswith('.txt') and index in f
    ]
    if not matches:
        return ""
    return (
        open(os.path.join(gt_dir, matches[0]), encoding='utf-8')
        .read()
        .strip()
        .replace('-e', '')
        .strip()
    )


ORIGINAL = {
    'scene_000.png': {'accuracy': 100.0, 'cer': 0.0,  'wer': 0.0},
    'scene_001.png': {'accuracy': 100.0, 'cer': 0.0,  'wer': 0.0},
    'scene_002.png': {'accuracy': 100.0, 'cer': 0.0,  'wer': 0.0},
    'scene_003.png': {'accuracy':   0.0, 'cer': 20.0, 'wer': 100.0},
    'scene_004.png': {'accuracy':   0.0, 'cer': 20.0, 'wer': 100.0},
    'scene_005.png': {'accuracy':   0.0, 'cer': 66.7, 'wer': 100.0},
}


if __name__ == "__main__":
    upscaled_dir = os.path.join(RESULTS, "scene_upscaled")
    gt_dir       = os.path.join(DATA, "scene_text", "ground_truth")

    print("=" * 60)
    print("  SCENE TEXT — BEFORE vs AFTER UPSCALING")
    print("=" * 60)

    model   = load_model()
    results = []

    for img_name in sorted(os.listdir(upscaled_dir)):
        if not img_name.endswith('.png'):
            continue

        img_path = os.path.join(upscaled_dir, img_name)
        gt_text  = read_ground_truth(gt_dir, img_name)

        ocr_text, confidence, _ = extract_text(model, img_path)

        acc = word_accuracy(ocr_text, gt_text)
        cer = calculate_cer(clean(ocr_text), clean(gt_text))
        wer = calculate_wer(ocr_text, gt_text)

        original = ORIGINAL.get(img_name, {})
        acc_diff = round(acc - original.get('accuracy', 0), 1)

        print(f"\n  Image      : {img_name}")
        print(f"  GT Text    : {gt_text}")
        print(f"  OCR Text   : {ocr_text}")
        print(f"  Accuracy   : {original.get('accuracy')}% → {acc}%  ({'+' if acc_diff > 0 else ''}{acc_diff}%)")
        print(f"  CER        : {original.get('cer')}% → {round(cer*100,1)}%")
        print(f"  WER        : {original.get('wer')}% → {round(wer*100,1)}%")
        print(f"  Confidence : {confidence}%")

        if acc_diff > 0:
            print(f"  Result     : IMPROVED")
        elif acc_diff == 0:
            print(f"  Result     : NO CHANGE")
        else:
            print(f"  Result     : WORSE")

        results.append({
            'image'           : img_name,
            'gt_text'         : gt_text,
            'ocr_text'        : ocr_text,
            'accuracy_before' : original.get('accuracy'),
            'accuracy_after'  : acc,
            'accuracy_change' : acc_diff,
            'confidence'      : confidence
        })

    # save comparison
    csv_path = os.path.join(RESULTS, "scene_comparison.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'image', 'gt_text', 'ocr_text',
            'accuracy_before', 'accuracy_after',
            'accuracy_change', 'confidence'
        ])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n  Comparison saved: {csv_path}")
    print("\n  Done.")