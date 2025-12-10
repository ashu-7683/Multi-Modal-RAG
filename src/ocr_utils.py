# src/ocr_utils.py
from pytesseract import image_to_string
from PIL import Image

def run_ocr_on_image(pil_image: Image.Image) -> str:
    gray = pil_image.convert("L")
    text = image_to_string(gray, lang='eng')
    return text
