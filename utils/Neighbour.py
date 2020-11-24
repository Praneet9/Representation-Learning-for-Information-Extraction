import json
import traceback


def attach_neighbour(annotation, ocr_path):

    for anno in annotation:
        try:
            file_name = anno['filename']
            ocr_json = ocr_path / (file_name + ".json")
            with open(ocr_json, 'r') as f:
                ocr_data = json.load(f)

            print()
        except Exception:
            trace = traceback.format_exc()
            print("Error in finding neighbour: %s : %s" % (anno['filename'], trace))

    return annotation