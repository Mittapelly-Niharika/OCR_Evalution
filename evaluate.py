# evaluate.py
# Main file. Runs the full evaluation pipeline.
#
# What it does:
#   1. Loads PaddleOCR model once
#   2. Goes through all 6 document categories
#   3. For each image runs OCR and calculates metrics
#   4. Saves results to CSV files
#   5. Prints final summary

import os
import csv

from config     import DATA, RESULTS, CATEGORIES
from ocr_engine import load_model, extract_text
from metrics    import (
    word_accuracy,
    calculate_cer,
    calculate_wer,
    missing_words,
    failure_reason,
    clean
)


# Read ground truth file for one image

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


# Process one category

def run_category(model, category):
    img_dir = os.path.join(DATA, category, "images")
    gt_dir  = os.path.join(DATA, category, "ground_truth")

    print(f"\n{'='*55}")
    print(f"  {category.upper()}")
    print(f"{'='*55}")

    results   = []
    extracted = []

    for image_name in sorted(os.listdir(img_dir)):
        if not image_name.endswith('.png'):
            continue

        image_path   = os.path.join(img_dir, image_name)
        gt_text      = read_ground_truth(gt_dir, image_name)
        ocr_text, confidence, _ = extract_text(model, image_path)

        acc     = word_accuracy(ocr_text, gt_text)
        cer     = calculate_cer(clean(ocr_text), clean(gt_text))
        wer     = calculate_wer(ocr_text, gt_text)
        missing = missing_words(ocr_text, gt_text)
        reasons = failure_reason(missing, category)

        print(f"\n  Image      : {image_name}")
        print(f"  Accuracy   : {acc}%")
        print(f"  CER        : {round(cer*100, 1)}%")
        print(f"  WER        : {round(wer*100, 1)}%")
        print(f"  Confidence : {confidence}%")
        print(f"  Missing    : {len(missing)} words")

        if missing:
            print(f"  Examples   : {', '.join(missing[:5])}")

        if reasons:
            print(f"  Why failing:")
            for r in reasons:
                print(f"    - {r}")

        results.append({
            'image'     : image_name,
            'accuracy'  : acc,
            'cer'       : round(cer*100, 1),
            'wer'       : round(wer*100, 1),
            'confidence': confidence,
            'missing'   : len(missing)
        })

        extracted.append({
            'image'   : image_name,
            'gt_text' : gt_text,
            'ocr_text': ocr_text,
            'match'   : 'YES' if acc >= 80 else 'NO'
        })

    if not results:
        return None

    avg_acc = round(sum(r['accuracy'] for r in results) / len(results), 1)
    avg_cer = round(sum(r['cer']      for r in results) / len(results), 1)
    avg_wer = round(sum(r['wer']      for r in results) / len(results), 1)

    # save metrics CSV
    metrics_path = os.path.join(RESULTS, f"{category}_results.csv")
    with open(metrics_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'image', 'accuracy', 'cer', 'wer', 'confidence', 'missing'
        ])
        writer.writeheader()
        writer.writerows(results)
        writer.writerow({
            'image'     : 'AVERAGE',
            'accuracy'  : avg_acc,
            'cer'       : avg_cer,
            'wer'       : avg_wer,
            'confidence': '',
            'missing'   : ''
        })

    # save extracted text CSV
    extracted_path = os.path.join(RESULTS, f"{category}_extracted.csv")
    with open(extracted_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'image', 'gt_text', 'ocr_text', 'match'
        ])
        writer.writeheader()
        writer.writerows(extracted)

    print(f"\n  Average: Accuracy={avg_acc}%  CER={avg_cer}%  WER={avg_wer}%")
    print(f"  Saved: {metrics_path}")

    return {
        'category': category,
        'accuracy': avg_acc,
        'cer'     : avg_cer,
        'wer'     : avg_wer
    }


# Main

if __name__ == "__main__":
    model   = load_model()
    summary = []

    for category in CATEGORIES:
        result = run_category(model, category)
        if result:
            summary.append(result)

    # print final summary
    print(f"\n{'='*55}")
    print("  FINAL SUMMARY")
    print(f"{'='*55}")
    print(f"\n  {'Category':<15} {'Accuracy':>9} {'CER':>6} {'WER':>6}")
    print(f"  {'-'*40}")
    for r in summary:
        print(
            f"  {r['category']:<15}"
            f"{r['accuracy']:>8}%"
            f"{r['cer']:>6}%"
            f"{r['wer']:>6}%"
        )

    # save final summary CSV
    summary_path = os.path.join(RESULTS, "final_summary.csv")
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'category', 'accuracy', 'cer', 'wer'
        ])
        writer.writeheader()
        writer.writerows(summary)

    print(f"\n  Summary saved: {summary_path}")
    print("\n  Done.")