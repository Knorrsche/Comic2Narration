from PIL import Image
import numpy as np
import fitz
import xml.etree.ElementTree as eT
from xml.dom import minidom
import pathlib
import tempfile
import pyttsx3
from gtts import gTTS
import os

from src.Classes import PageType, SpeechBubbleType, Series, Comic, Page, Panel, Entity, EntityTemplate, SpeechBubble
from src.Utils.ImageUtils import image_from_bbox


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


# TODO: make universal export and import functions
def save_xml_to_file(filepath, xml_str):
    with open(filepath, 'w') as file:
        file.write(xml_str)

# TODO: currently, the text is only getting processed after export to xml, add process before
def save_script_as_txt(filepath, script):
    with open(filepath, 'w') as file:
        file.write(script)


def save_script_as_mp3(filepath, script):
    #tts = gTTS(text=script, lang='en')
    #tts.save(filepath)
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice',voices[1].id )
    engine.save_to_file(script, filepath)
    engine.runAndWait()
    print(f'Created MP3 file: {filepath}')

def read_xml_from_pdf(pdf_path: str):
    doc = fitz.open(pdf_path)

    if doc.embfile_count() <= 0:
        raise Exception("No embedded XML found")

    return doc.embfile_get(0)


# TODO: Add error handling if embedding already exists
def add_annotation_to_pdf(pdf_path: str, xml_content: str, new_pdf_path: str):
    doc = fitz.open(pdf_path)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".xml") as temp_xml_file:
        temp_xml_file.write(xml_content.encode('utf-8'))
        temp_xml_file_path = temp_xml_file.name

    try:
        embedded_data = pathlib.Path(temp_xml_file_path).read_bytes()

        doc.embfile_add(name="comic_data.xml", buffer_=embedded_data)

        doc.save(new_pdf_path)
    finally:
        doc.close()
        pathlib.Path(temp_xml_file_path).unlink()


def parse_bounding_box(bbox_str):
    parts = bbox_str.split(',')

    bbox_dict = {}

    for part in parts:
        key_value_pair = part.split(':')
        key = key_value_pair[0]
        value = ':'.join(key_value_pair[1:])
        if key in ['x', 'y', 'width', 'height', 'confidence']:
            bbox_dict[key] = float(value)
        else:
            bbox_dict[key] = value

    return bbox_dict


def parse_speech_bubble(sb_elem):
    return SpeechBubble(
        type=SpeechBubbleType[sb_elem.find('Type').text.upper()],
        text=sb_elem.find('Text').text,
        bounding_box=parse_bounding_box(sb_elem.find('BoundingBox').text)
    )


def parse_panel(panel_elem):
    speech_bubbles = [parse_speech_bubble(sb) for sb in panel_elem.find('SpeechBubbles')]
    return Panel(
        description=panel_elem.find('Description').text,
        bounding_box=parse_bounding_box(panel_elem.find('BoundingBox').text),
        speech_bubbles=speech_bubbles
    )


def parse_page(page_elem):
    panels = [parse_panel(panel) for panel in page_elem.find('Panels')]
    return Page(
        page_index=int(page_elem.find('Index').text),
        page_type=PageType[page_elem.find('Type').text.upper()],
        panels=panels
    )


def parse_page_pair(page_pair_elem):
    right_page = parse_page(page_pair_elem.find('RightPage/Page')) if page_pair_elem.find('RightPage') else None
    left_page = parse_page(page_pair_elem.find('LeftPage/Page')) if page_pair_elem.find('LeftPage') else None
    return left_page, right_page


def parse_series(series_elem):
    name_elem = series_elem.find('Name')
    if name_elem is not None:
        return Series(name=name_elem.text)
    else:
        return Series(name='')


# TODO: add image data
def parse_comic(xml_content):
    root = eT.fromstring(xml_content)
    name = root.find('Name').text
    volume = root.find('Volume').text
    main_series = parse_series(root.find('MainSeries'))
    secondary_series = [parse_series(series) for series in root.find('SecondarySeries')]
    page_pairs = [parse_page_pair(pp) for pp in root.find('PagePairs')]

    return Comic(name, volume, main_series, secondary_series, page_pairs)


def add_image_data(comic: Comic, file_path: str):
    pages = convert_pdf_to_image(file_path)
    counter = 0
    for page_pair in comic.page_pairs:

        left_page, right_page = page_pair

        if left_page is not None:
            left_page.page_image = pages[counter]
            for panel in left_page.panels:

                panel.image = image_from_bbox(left_page.page_image, panel.bounding_box)

                for speech_bubble in panel.speech_bubbles:
                    speech_bubble.image = image_from_bbox(left_page.page_image, speech_bubble.bounding_box)
            counter += 1

        if right_page is not None:
            right_page.page_image = pages[counter]
            for panel in right_page.panels:
                panel.image = image_from_bbox(right_page.page_image, panel.bounding_box)

                for speech_bubble in panel.speech_bubbles:
                    speech_bubble.image = image_from_bbox(right_page.page_image, speech_bubble.bounding_box)
            counter += 1
