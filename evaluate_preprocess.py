# evaluate_preprocessed.py
# Runs OCR on preprocessed degraded images
# and compares results with original OCR results.
#
# Goal: see how much preprocessing improved accuracy.

import os
import csv

from config     import DATA, RESULTS
from ocr_engine import load_model, extract_text
from metrics    import (
    word_accuracy,
    calculate_cer,
    calculate_wer,
    missing_words,
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


# Original results from evaluate.py
# We use these to compare before vs after
ORIGINAL = {
    'degraded_000.png': {'accuracy': 89.7, 'cer': 12.7, 'wer': 22.1},
    'degraded_001.png': {'accuracy':  0.0, 'cer': 98.5, 'wer': 100.0},
    'degraded_002.png': {'accuracy': 85.7, 'cer': 32.2, 'wer': 45.2},
    'degraded_003.png': {'accuracy': 50.7, 'cer': 75.3, 'wer': 96.8},
    'degraded_004.png': {'accuracy': 52.2, 'cer': 43.5, 'wer': 75.6},
    'degraded_005.png': {'accuracy':  0.0, 'cer': 98.9, 'wer': 100.0},
}


if __name__ == "__main__":
    processed_dir = os.path.join(RESULTS, "degraded_processed")
    gt_dir        = os.path.join(DATA, "degraded", "ground_truth")

    print("=" * 60)
    print("  DEGRADED — BEFORE vs AFTER PREPROCESSING")
    print("=" * 60)

    model   = load_model()
    results = []

    for img_name in sorted(os.listdir(processed_dir)):
        if not img_name.endswith('.png'):
            continue

        img_path = os.path.join(processed_dir, img_name)
        gt_text  = read_ground_truth(gt_dir, img_name)

        ocr_text, confidence, _ = extract_text(model, img_path)

        acc = word_accuracy(ocr_text, gt_text)
        cer = calculate_cer(clean(ocr_text), clean(gt_text))
        wer = calculate_wer(ocr_text, gt_text)

        original = ORIGINAL.get(img_name, {})
        acc_diff = round(acc - original.get('accuracy', 0), 1)
        cer_diff = round(cer*100 - original.get('cer', 0), 1)
        wer_diff = round(wer*100 - original.get('wer', 0), 1)

        print(f"\n  Image      : {img_name}")
        print(f"  Accuracy   : {original.get('accuracy')}% → {acc}%  ({'+' if acc_diff > 0 else ''}{acc_diff}%)")
        print(f"  CER        : {original.get('cer')}%  → {round(cer*100,1)}%  ({'+' if cer_diff > 0 else ''}{cer_diff}%)")
        print(f"  WER        : {original.get('wer')}% → {round(wer*100,1)}%  ({'+' if wer_diff > 0 else ''}{wer_diff}%)")
        print(f"  Confidence : {confidence}%")

        if acc_diff > 0:
            print(f"  Result     : IMPROVED")
        elif acc_diff == 0:
            print(f"  Result     : NO CHANGE")
        else:
            print(f"  Result     : WORSE")

        results.append({
            'image'           : img_name,
            'accuracy_before' : original.get('accuracy'),
            'accuracy_after'  : acc,
            'accuracy_change' : acc_diff,
            'cer_before'      : original.get('cer'),
            'cer_after'       : round(cer*100, 1),
            'wer_before'      : original.get('wer'),
            'wer_after'       : round(wer*100, 1),
            'confidence'      : confidence
        })

    # save comparison CSV
    csv_path = os.path.join(RESULTS, "degraded_comparison.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'image',
            'accuracy_before', 'accuracy_after', 'accuracy_change',
            'cer_before', 'cer_after',
            'wer_before', 'wer_after',
            'confidence'
        ])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n  Comparison saved: {csv_path}")
    print("\n  Done.")