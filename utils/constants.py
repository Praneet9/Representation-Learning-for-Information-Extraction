from pathlib import Path


NEIGHBOURS = 5
HEADS = 4
EMBEDDING_SIZE = 16
VOCAB_SIZE = 1000
BATCH_SIZE = 2
EPOCHS = 10
VAL_SPLIT = 0.2
LR = 0.001

current_directory = Path.cwd()
XMLS = current_directory / "dataset" / "xmls"
OCR = current_directory / "dataset" / "tesseract_results_lstm"
IMAGES = current_directory / "dataset" / "images"
CANDIDATES = current_directory / "dataset" / "candidates"
OUTPUT_PATH = current_directory / "output"
FIELDS = {'invoice_date': 0, 'invoice_no': 1, 'total': 2}

if not OUTPUT_PATH.exists():
    OUTPUT_PATH.mkdir(parents=True)