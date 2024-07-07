from PIL import Image
import numpy as np
import fitz

def convert_pdf_to_image(pdf_path: str):
    pdf = fitz.open(pdf_path)

    rgb_arrays = []
    for page_number in range(len(pdf)):
        page = pdf.load_page(page_number)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB",[pix.width,pix.height],pix.samples)
        rgb_array = np.array(img)
        rgb_arrays.append(rgb_array)
     
    return rgb_arrays

