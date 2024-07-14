from enum import Enum
from typing import List, Optional
from .Entity import Entity
import xml.etree.ElementTree as ET
import xml.sax.saxutils as saxutils


class SpeechBubbleType(Enum):
    NARRATOR = 1
    SPEECH = 2
    THOUGHTS = 3
    EFFECT = 4
    SHOUT = 5


class SpeechBubble:

    def __init__(self, type: SpeechBubbleType, text: str, bounding_box, image=None):
        self.type: SpeechBubbleType = type
        self.text = text
        self.bounding_box = bounding_box
        self.image = image
        self.speaker: List[Entity] = []

    def set_speaker(self, speaker: List[Entity]):
        self.speaker = speaker

    # TODO: Create universal function to parse strings to right format
    def to_xml(self):
        element = ET.Element('SpeechBubble')
        ET.SubElement(element, 'Type').text = self.type.name
        ET.SubElement(element, 'Text').text = self.escape_text(self.text) if self.text else ''
        bbox = ET.SubElement(element, 'BoundingBox')
        bbox.text = ','.join(f"{key}:{value}" for key, value in self.bounding_box.items())

        speakers_element = ET.SubElement(element, 'Speakers')
        for entity in self.speaker:
            speakers_element.append(entity.to_xml())

        return element

    # TODO: refactor and add special char dict and add reverse
    def escape_text(self,text):
        text = text.encode('utf-8').decode('utf-8')

        escaped_text = saxutils.escape(text)

        xml_friendly_text = escaped_text.replace('\n', ' ')

        special_chars = {
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

        for char, entity in special_chars.items():
            xml_friendly_text = xml_friendly_text.replace(char, entity)

        return xml_friendly_text
