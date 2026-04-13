# config.py
# All project settings in one place.
# If you move the project, only change this file.

import os

# Root folder of the project
BASE = os.path.dirname(os.path.abspath(__file__))

# Data folder containing all 6 document types
DATA = os.path.join(BASE, "Data")

# Results folder where all output files are saved
RESULTS = os.path.join(BASE, "results")

# The 6 document categories we are evaluating
CATEGORIES = [
    "degraded",
    "printed",
    "dense_text",
    "handwritten",
    "receipts",
    "scene_text"
]

# Confidence threshold
# Above 85  = accepted, output is reliable
# 60 to 85  = needs review, output may have errors
# Below 60  = low confidence, do not trust output
HIGH_CONFIDENCE = 85
LOW_CONFIDENCE  = 60

# Create results folder if it does not exist
os.makedirs(RESULTS, exist_ok=True)