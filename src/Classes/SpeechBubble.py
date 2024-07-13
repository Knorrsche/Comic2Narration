from enum import Enum
from typing import List
from .Entity import Entity
import xml.etree.ElementTree as ET


class SpeechBubbleType(Enum):
    NARRATOR = 1
    SPEECH = 2
    THOUGHTS = 3
    EFFECT = 4
    SHOUT = 5


class SpeechBubble:

    def __init__(self, type: SpeechBubbleType, text: str, bounding_box, image):
        self.type: SpeechBubbleType = type
        self.text = text
        self.bounding_box = bounding_box
        self.image = image
        self.speaker: List[Entity] = []

    # TODO: remove if image data added to import
    def __init__(self, type: SpeechBubbleType, text: str, bounding_box):
        self.type: SpeechBubbleType = type
        self.text = text
        self.bounding_box = bounding_box
        self.speaker: List[Entity] = []

    def set_speaker(self, speaker: List[Entity]):
        self.speaker = speaker

    def to_xml(self):
        element = ET.Element('SpeechBubble')
        ET.SubElement(element, 'Type').text = self.type.name
        ET.SubElement(element, 'Text').text = self.text
        bbox = ET.SubElement(element, 'BoundingBox')
        bbox.text = ','.join(f"{key}:{value}" for key, value in self.bounding_box.items())

        speakers_element = ET.SubElement(element, 'Speakers')
        for entity in self.speaker:
            speakers_element.append(entity.to_xml())

        return element
