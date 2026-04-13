## Dataset

Dataset is included in the repository under the `Data/` folder.
6 document categories — printed, degraded, dense text, 
handwritten, receipts, scene text.
36 images total with ground truth annotations.

# PaddleOCR Evaluation

Systematic evaluation of PaddleOCR 3.4.0 across 6 real-world 
document categories — printed, dense text, receipts, degraded, 
scene text, and handwritten.

## Results Summary

| Category    | Accuracy | CER   | WER   |
|-------------|----------|-------|-------|
| Receipts    | 97.1%    | 18.6% | 26.5% |
| Dense Text  | 85.6%    | 29.2% | 44.6% |
| Printed     | 82.0%    | 41.4% | 58.1% |
| Scene Text  | 50.0%    | 17.8% | 50.0% |
| Degraded    | 46.4%    | 60.2% | 73.3% |
| Handwritten | 14.4%    | 70.9% | 93.0% |

## Setup

```bash
py -3.12 -m venv venv --without-pip
venv\Scripts\python.exe get-pip.py
venv\Scripts\pip.exe install -r requirements.txt
```

## Run Evaluation

```bash
python evaluate.py
```

## Project Structure
