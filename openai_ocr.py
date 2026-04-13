# openai_ocr.py
# Uses OpenAI GPT-4 Vision to extract text from images.
#
# Why GPT-4 Vision for handwritten:
#   PaddleOCR scored 14.4% on handwritten documents.
#   GPT-4 Vision is trained on all types of images
#   including handwriting, cursive, old documents.
#   It understands context which helps with unclear letters.

import os
import csv
import base64
from openai import OpenAI

from config  import DATA, RESULTS
from metrics import (
    word_accuracy,
    calculate_cer,
    calculate_wer,
    clean
)


# Convert image to base64
# OpenAI API accepts images as base64 encoded strings

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# Extract text using GPT-4 Vision

def openai_extract(client, image_path):
    base64_image = encode_image(image_path)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Read all the text in this image exactly as it appears. Return only the text. Do not add any explanation."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=500
    )

    return response.choices[0].message.content.strip()


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


# Original PaddleOCR results for comparison
ORIGINAL = {
    'handwritten_000.png': {'accuracy': 14.3, 'cer': 50.0,  'wer': 85.7},
    'handwritten_001.png': {'accuracy': 11.1, 'cer': 69.7,  'wer': 88.9},
    'handwritten_002.png': {'accuracy': 50.0, 'cer': 70.3,  'wer': 83.3},
    'handwritten_003.png': {'accuracy':  0.0, 'cer': 57.6,  'wer': 100.0},
    'handwritten_004.png': {'accuracy':  0.0, 'cer': 100.0, 'wer': 100.0},
    'handwritten_005.png': {'accuracy': 11.1, 'cer': 77.8,  'wer': 100.0},
}


if __name__ == "__main__":

    # Get API key from environment variable
    API_KEY = os.getenv("OPENAI_API_KEY")
    if not API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set. Please set it before running this script.")

    client  = OpenAI(api_key=API_KEY)

    img_dir = os.path.join(DATA, "handwritten", "images")
    gt_dir  = os.path.join(DATA, "handwritten", "ground_truth")

    print("=" * 60)
    print("  HANDWRITTEN — PaddleOCR vs GPT-4 Vision")
    print("=" * 60)

    results = []

    for img_name in sorted(os.listdir(img_dir)):
        if not img_name.endswith('.png'):
            continue

        img_path = os.path.join(img_dir, img_name)
        gt_text  = read_ground_truth(gt_dir, img_name)

        try:
            openai_text = openai_extract(client, img_path)
        except Exception as e:
            print(f"  OpenAI failed on {img_name}: {e}")
            openai_text = ""

        acc = word_accuracy(openai_text, gt_text)
        cer = calculate_cer(clean(openai_text), clean(gt_text))
        wer = calculate_wer(openai_text, gt_text)

        original = ORIGINAL.get(img_name, {})
        acc_diff = round(acc - original.get('accuracy', 0), 1)

        print(f"\n  Image          : {img_name}")
        print(f"  GT Text        : {gt_text}")
        print(f"  OpenAI Text    : {openai_text}")
        print(f"  PaddleOCR Acc  : {original.get('accuracy')}%")
        print(f"  OpenAI Acc     : {acc}%  ({'+' if acc_diff > 0 else ''}{acc_diff}%)")
        print(f"  CER            : {original.get('cer')}% → {round(cer*100,1)}%")
        print(f"  WER            : {original.get('wer')}% → {round(wer*100,1)}%")

        if acc_diff > 0:
            print(f"  Result         : IMPROVED")
        elif acc_diff == 0:
            print(f"  Result         : NO CHANGE")
        else:
            print(f"  Result         : WORSE")

        results.append({
            'image'           : img_name,
            'gt_text'         : gt_text,
            'openai_text'     : openai_text,
            'paddle_accuracy' : original.get('accuracy'),
            'openai_accuracy' : acc,
            'improvement'     : acc_diff,
            'cer_before'      : original.get('cer'),
            'cer_after'       : round(cer*100, 1),
            'wer_before'      : original.get('wer'),
            'wer_after'       : round(wer*100, 1)
        })

    # summary
    paddle_avg = round(
        sum(r['paddle_accuracy'] for r in results) / len(results), 1
    )
    openai_avg = round(
        sum(r['openai_accuracy'] for r in results) / len(results), 1
    )

    print(f"\n  {'='*55}")
    print(f"  PaddleOCR Average : {paddle_avg}%")
    print(f"  OpenAI Average    : {openai_avg}%")
    print(f"  Improvement       : +{round(openai_avg - paddle_avg, 1)}%")

    # save CSV
    csv_path = os.path.join(RESULTS, "handwritten_comparison.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'image', 'gt_text', 'openai_text',
            'paddle_accuracy', 'openai_accuracy', 'improvement',
            'cer_before', 'cer_after',
            'wer_before', 'wer_after'
        ])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n  Saved: {csv_path}")
    print("\n  Done.")