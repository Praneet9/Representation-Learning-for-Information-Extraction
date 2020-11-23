from pathlib import Path
from utils import xml_parser

this_dir = Path.cwd()
xmls_path = this_dir / "dataset" / "xmls"


annotation, classes_count, class_mapping = xml_parser.get_data(xmls_path)



print()

