# metrics.py
# Calculates how good or bad the OCR output is
# by comparing it with the ground truth text.
#
# Three metrics are used:
#
# Accuracy   - what percentage of ground truth words OCR found
# CER        - how many characters are wrong (lower is better)
# WER        - how many words are wrong (lower is better)
#
# All three together tell the full story.
# Accuracy alone is not enough.

import re


# Clean text before comparing
# Why: OCR returns "Hello," and ground truth has "hello"
# Without cleaning these look different but mean the same
# So we lowercase and remove punctuation first

def clean(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Word Accuracy
# Simple count - how many ground truth words did OCR find
#
# Example:
#   ground truth : invoice total amount due   (4 words)
#   ocr output   : invoice total amount        (3 words)
#   accuracy     : 3/4 = 75%

def word_accuracy(ocr_text, gt_text):
    gt_words  = clean(gt_text).split()
    ocr_words = set(clean(ocr_text).split())

    if not gt_words:
        return 0.0

    matched = sum(1 for word in gt_words if word in ocr_words)
    return round(matched / len(gt_words) * 100, 1)


# CER - Character Error Rate
# Counts minimum character edits to convert OCR text to ground truth
# Each edit is one substitution, insertion or deletion
#
# Example:
#   ground truth : hello world    (11 characters)
#   ocr output   : helo  world    (1 wrong character)
#   CER          : 1/11 = 9%
#
# Uses Levenshtein distance algorithm
# We wrote this ourselves - no external library needed

def calculate_cer(ocr_text, gt_text):
    o = ocr_text.lower().strip()
    g = gt_text.lower().strip()

    if not g:
        return 0.0

    rows = len(g) + 1
    cols = len(o) + 1
    m    = [[0] * cols for _ in range(rows)]

    for i in range(rows): m[i][0] = i
    for j in range(cols): m[0][j] = j

    for i in range(1, rows):
        for j in range(1, cols):
            if g[i-1] == o[j-1]:
                m[i][j] = m[i-1][j-1]
            else:
                m[i][j] = 1 + min(
                    m[i-1][j],
                    m[i][j-1],
                    m[i-1][j-1]
                )

    return round(m[rows-1][cols-1] / len(g), 4)


# WER - Word Error Rate
# Same as CER but counts wrong words instead of characters
#
# Example:
#   ground truth : invoice total amount   (3 words)
#   ocr output   : invoice totol amount   (1 wrong word)
#   WER          : 1/3 = 33%

def calculate_wer(ocr_text, gt_text):
    o = clean(ocr_text).split()
    g = clean(gt_text).split()

    if not g:
        return 0.0

    rows = len(g) + 1
    cols = len(o) + 1
    m    = [[0] * cols for _ in range(rows)]

    for i in range(rows): m[i][0] = i
    for j in range(cols): m[0][j] = j

    for i in range(1, rows):
        for j in range(1, cols):
            if g[i-1] == o[j-1]:
                m[i][j] = m[i-1][j-1]
            else:
                m[i][j] = 1 + min(
                    m[i-1][j],
                    m[i][j-1],
                    m[i-1][j-1]
                )

    return round(m[rows-1][cols-1] / len(g), 4)


# Find missing words
# Words that are in ground truth but OCR did not detect
# These are the actual failures

def missing_words(ocr_text, gt_text):
    gt  = set(clean(gt_text).split())
    ocr = set(clean(ocr_text).split())
    return sorted(gt - ocr)


# Failure reason
# Explains why OCR failed on this specific image
# This is what your manager wants - not just numbers
# but actual reasons

def failure_reason(missing, category):
    reasons = []

    if any(re.search(r'\d', w) for w in missing):
        reasons.append(
            "Numbers missed - OCR confuses 0/O, 1/l, 8/B"
        )

    if any(len(w) <= 2 for w in missing):
        reasons.append(
            "Short words missed - small text regions get skipped"
        )

    category_notes = {
        "degraded"   : "Faded ink and noise reduce contrast. "
                       "Text detection fails when text and background "
                       "have similar brightness.",
        "handwritten": "PaddleOCR is trained on printed text only. "
                       "Cursive handwriting looks completely different "
                       "to the model.",
        "scene_text" : "Very small image resolution. "
                       "PaddleOCR needs minimum size to detect text.",
        "dense_text" : "Tightly packed text causes word merging. "
                       "Adjacent words detected as one long word.",
        "receipts"   : "Special characters in codes like IBAN and Tax ID "
                       "get stripped during comparison.",
        "printed"    : "Special characters like R&D and B&W "
                       "get stripped during text cleaning."
    }

    if category in category_notes:
        reasons.append(category_notes[category])

    return reasons