from pathlib import Path


NEIGHBOURS = 5
HEADS = 4
EMBEDDING_SIZE = 16
VOCAB_SIZE = 1000
BATCH_SIZE = 2
EPOCHS = 10
LR = 0.001

current_directory = Path.cwd()
XML_DIR = current_directory / "dataset" / "xmls"
OCR_DIR = current_directory / "dataset" / "tesseract_results_lstm"
IMAGE_DIR = current_directory / "dataset" / "images"
CANDIDATE_DIR = current_directory / "dataset" / "candidates"
SPLIT_DIR = current_directory / "dataset" / "split"
OUTPUT_DIR = current_directory / "output"

if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir(parents=True)