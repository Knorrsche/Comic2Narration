from typing import Tuple, List
from .Entity import Entity
from .SpeechBubble import SpeechBubble
import xml.etree.ElementTree as ET


class Panel:
    def __init__(self, description: str, bounding_box, image=None, speech_bubbles=None):
        self.description = description
        self.bounding_box = bounding_box
        self.image = image
        self.starting_tag = False
        self.scene_id = 0
        self.entities: List[Entity] = []
        self.descriptions = []
        #TODO: Update import export wiht page_id
        self.page_id = 0

        if speech_bubbles is None:
            self.speech_bubbles: List[SpeechBubble] = []
        else:
            self.speech_bubbles: List[SpeechBubble] = speech_bubbles

    def get_transcript(self):
        transcript = ''
        for i, speech_bubble in enumerate(self.speech_bubbles):
            transcript += f'Speech Bubble {i + 1}: {speech_bubble.text}/n'
        transcript += '\n'
        return transcript

    def to_xml(self):
        element = ET.Element('Panel')

        ET.SubElement(element, 'Description').text = self.description
        ET.SubElement(element,'Scene_Id').text = str(self.scene_id)
        ET.SubElement(element, 'Starting_Tag').text = str(self.starting_tag)

        bbox = ET.SubElement(element, 'BoundingBox')
        bbox.text = ','.join(f"{key}:{value}" for key, value in self.bounding_box.items())

        entities_element = ET.SubElement(element, 'Entities')
        for entity in self.entities:
            entities_element.append(entity.to_xml())

        speech_bubbles_element = ET.SubElement(element, 'SpeechBubbles')
        for speech_bubble in self.speech_bubbles:
            speech_bubbles_element.append(speech_bubble.to_xml())

        return element
