# ocr_engine.py
# Responsible for loading PaddleOCR and extracting text from images.
#
# Why a separate file?
# If we switch to a different OCR engine later,
# we only change this file. Nothing else breaks.

import os
from paddleocr import PaddleOCR

# Disable the slow internet connectivity check on startup
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"


# Load PaddleOCR once
# lang='en' means English model
# We load it once and reuse it for all 36 images
# Loading takes ~30 seconds so we never load it twice

def load_model():
    print("Loading PaddleOCR model...")
    model = PaddleOCR(lang='en')
    print("Model ready.")
    return model


# Extract text from one image
# Returns three things:
#   text       → the full extracted text as one string
#   confidence → average confidence across all detected words
#   words      → list of individual detected words

def extract_text(model, image_path):
    result = list(model.predict(image_path))

    # If nothing was detected return empty values
    if not result or not isinstance(result[0], dict):
        return "", 0.0, []

    words  = result[0].get("rec_texts", [])
    scores = result[0].get("rec_scores", [])

    # If no words detected return empty values
    if not words:
        return "", 0.0, []

    text       = " ".join(words)
    confidence = round(sum(scores) / len(scores) * 100, 1)

    return text, confidence, words