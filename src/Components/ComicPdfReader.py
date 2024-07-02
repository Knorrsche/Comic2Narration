from pdf2image import convert_from_path
from PIL import Image
import numpy as np
from Classes.Page import Page,PageType

def convert_pdf_to_image(pdf_path: str):
    images = convert_from_path(pdf_path, poppler_path=r"c:\Users\derra\Downloads\Release-24.02.0-0\poppler-24.02.0\Library\bin")

    rgb_arrays = []
    for image in images:
        rgb_image = image.convert('RGB')
        rgb_array = np.array(rgb_image)
        rgb_arrays.append(rgb_array)

    return rgb_arrays

