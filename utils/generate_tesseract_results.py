from pytesseract import Output
import pytesseract


def get_tesseract_results(image):
    ocr_result = pytesseract.image_to_data(image, config='--oem 1 --psm 3', output_type=Output.DICT)
    return ocr_result