from PIL import Image
import numpy as np
import fitz
import xml.etree.ElementTree as eT
import xml.sax.saxutils as saxutils

from xml.dom import minidom
import pathlib
import tempfile
import pyttsx3
from gtts import gTTS
import os

from src.Classes import PageType, SpeechBubbleType, Series, Comic, Page, Panel, Entity, EntityTemplate, SpeechBubble
from src.Utils.ImageUtils import image_from_bbox


def escape_text(text, reverse=False):
    if text is None: return text

    text = text.encode('utf-8').decode('utf-8')

    escaped_text = saxutils.escape(text)

    xml_friendly_text = escaped_text.replace('\n', ' ')

    special_chars = {
        '&': '&amp;',
        '©': '&copy;',
        '®': '&reg;',
        '™': '&trade;',
        '€': '&euro;',
        '£': '&pound;',
        '’': '&rsquo;',
        '‘': '&lsquo;',
        '“': '&ldquo;',
        '”': '&rdquo;',
        '|': '&#124;',
        '—': '&mdash;',
        '–': '&ndash;',
        '»': '&raquo;',
        '¥': '&yen;',
        '«': '&laquo;',
        '¢': '&cent;'
    }

    if reverse:
        for char, entity in special_chars.items():
            xml_friendly_text = xml_friendly_text.replace(entity, char)
    else:
        for char, entity in special_chars.items():
            xml_friendly_text = xml_friendly_text.replace(char, entity)

    return xml_friendly_text


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


# TODO: currently, the text is only getting processed after export to xml, add process before
def save_script_as_txt(filepath, script):
    with open(filepath, 'w') as file:
        file.write(script)


def save_script_as_mp3(filepath, script):
    #tts = gTTS(text=script, lang='en')
    #tts.save(filepath)
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
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
            try:
                bbox_dict[key] = float(value) if value else 0.0
            except ValueError as e:
                raise ValueError(f"Invalid value for {key}: {value}") from e
        else:
            bbox_dict[key] = value

    return bbox_dict


def parse_speech_bubble(sb_elem):
    speech_bubble = SpeechBubble(
        type=sb_elem.find('Type').text,
        text=escape_text(sb_elem.find('Text').text, reverse=True),
        bounding_box=parse_bounding_box(sb_elem.find('BoundingBox').text)
    )
    speech_bubble.speaker_id = int(sb_elem.find('Speaker_Id').text)
    return speech_bubble


def parase_entities(en_elem):
    entity = Entity(
        bounding_box=parse_bounding_box(en_elem.find('BoundingBox').text),
    )
    entity.named_entity_id = int(escape_text(en_elem.find('Named_Entity_Id').text))
    entity.tags = parse_tags(en_elem.find('Tags'))
    entity.starting_tag = str_to_bool(en_elem.find('Active_Tag').text)
    return entity


def parse_tags(en_elem):
    tags = en_elem.findall('Tag')
    tag_list = []

    for tag in tags:
        label = tag.find('Label').text
        value = float(tag.find('Value').text)
        tag_list.append((label, value))

    return tag_list


def parse_panel(panel_elem):
    speech_bubbles = [parse_speech_bubble(sb) for sb in panel_elem.find('SpeechBubbles')]
    entities = [parase_entities(en) for en in panel_elem.find('Entities')]
    panel = Panel(
        description=panel_elem.find('Description').text,
        bounding_box=parse_bounding_box(panel_elem.find('BoundingBox').text),
        speech_bubbles=speech_bubbles,
    )
    panel.scene_id = int(panel_elem.find('Scene_Id').text)
    panel.starting_tag = str_to_bool(panel_elem.find('Starting_Tag').text)
    panel.entities = entities
    return panel


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


# TODO: add image data
def parse_comic(xml_content):
    root = eT.fromstring(xml_content)
    name = root.find('Name').text
    page_pairs = [parse_page_pair(pp) for pp in root.find('PagePairs')]

    return Comic(name, page_pairs)

def str_to_bool(s):
    if isinstance(s, str):
        s = s.strip().lower()
        if s == 'true':
            return True
        elif s == 'false':
            return False
    raise ValueError(f"Cannot convert {s} to boolean")


#TODO: add Taggs again?
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
                for entity in panel.entities:
                    entity.image = image_from_bbox(left_page.page_image, entity.bounding_box)
            counter += 1

        if right_page is not None:
            right_page.page_image = pages[counter]
            for panel in right_page.panels:
                panel.image = image_from_bbox(right_page.page_image, panel.bounding_box)

                for speech_bubble in panel.speech_bubbles:
                    speech_bubble.image = image_from_bbox(right_page.page_image, speech_bubble.bounding_box)
                for entity in panel.entities:
                    entity.image = image_from_bbox(right_page.page_image, entity.bounding_box)
            counter += 1
