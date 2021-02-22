from pathlib import Path
from random import shuffle

ROOT_DIR = Path.cwd()
SAVE_DIR = ROOT_DIR / 'dataset/split/'
ANNOTATION_DIR = ROOT_DIR / 'dataset/xmls/'

TRAIN_SPLIT_RATIO = 0.8

if not SAVE_DIR.exists():
    SAVE_DIR.mkdir(parents=True)

xml_files = [x.stem for x in ANNOTATION_DIR.glob("*.xml")]
shuffle(xml_files)
split_point = int(len(xml_files)*TRAIN_SPLIT_RATIO)

train_file = SAVE_DIR / "train.txt"
val_file = SAVE_DIR / "val.txt"

train_set = xml_files[:split_point]
val_set = xml_files[split_point:]

print("Files in train:", len(train_set))
print("Files in val:", len(val_set))

with open(train_file, 'w') as f:
    f.write("\n".join(train_set))

with open(val_file, 'w') as f:
    f.write("\n".join(val_set))


