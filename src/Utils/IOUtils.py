from PIL import Image
import numpy as np
import fitz
import xml.etree.ElementTree as ET
from enum import Enum
from xml.dom import minidom
import os


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

def prettify_xml(element):
    rough_string = ET.tostring(element, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def save_xml_to_file(filepath,xml_str):
    with open(filepath, 'w') as file:
        file.write(xml_str)

#TODO: make a better annotation method and format
def add_annotation_to_pdf(import_path,xml_str,export_path):
    doc = fitz.open(import_path)

    first_page = doc[0]  
    rect = fitz.Rect(100, 100, 200, 120)
    annot = first_page.add_text_annot(rect, "XML")
    
    annot.set_info(xml_str)
    annot.update()

    doc.save(export_path)
    doc.close()