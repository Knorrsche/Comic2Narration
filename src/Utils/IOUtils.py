from PIL import Image
import numpy as np
import fitz
import xml.etree.ElementTree as eT
from xml.dom import minidom
import os
import pathlib


def convert_pdf_to_image(pdf_path: str):
    pdf = fitz.open(pdf_path)

    rgb_arrays = []
    for page_number in range(len(pdf)):
        page = pdf.load_page(page_number)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        rgb_array = np.array(img)
        rgb_arrays.append(rgb_array)

    return rgb_arrays


def prettify_xml(element):
    rough_string = eT.tostring(element, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def save_xml_to_file(filepath, xml_str):
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

path = r"C:\Users\derra\Desktop\Bachelor"

namedoc = "Spiderman.pdf"

pathnamedoc = os.path.join(path, namedoc)

doc = fitz.open(pathnamedoc)

count = doc.embfile_count()
print("Number of embedded files before:", count)

namedata = "Spiderman.xml"
pathnamedata = os.path.join(path, namedata)
print("Path to XML file:", pathnamedata)
embedded_data = pathlib.Path(pathnamedata).read_bytes()
doc.embfile_add("Spiderman.xml", embedded_data)

namemp3 = "Spiderman.mp3"
pathnamemp3 = os.path.join(path, namemp3)
print("Path to MP3 file:", pathnamemp3)
embedded_mp3 = pathlib.Path(pathnamemp3).read_bytes()
doc.embfile_add("Spiderman.mp3", embedded_mp3)

doc.saveIncr()

doc.close()

doc = fitz.open(pathnamedoc)

count = doc.embfile_count()
print("Number of embedded files after:", count)

embedded_file_xml = doc.embfile_get(0)
if embedded_file_xml is not None:
    with open(os.path.join(path, "extracted_data.xml"), "wb") as f:
        f.write(embedded_file_xml)

embedded_file_mp3 = doc.embfile_get(1)
if embedded_file_mp3 is not None:
    with open(os.path.join(path, "extracted_audio.mp3"), "wb") as f:
        f.write(embedded_file_mp3)

doc.close()