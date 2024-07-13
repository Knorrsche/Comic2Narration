from typing import Tuple,List
from .Entity import Entity
from .SpeechBubble import SpeechBubble
import xml.etree.ElementTree as ET 

class Panel:
    def __init__(self,description: str, bounding_box,image = None):
        self.description = description
        self.bounding_box = bounding_box
        self.image = image
        self.entities: List[Entity] = []
        self.speech_bubbles: List[SpeechBubble]= []


    def to_xml(self):
        element = ET.Element('Panel')

        ET.SubElement(element, 'Description').text = self.description

        bbox = ET.SubElement(element, 'BoundingBox')
        bbox.text = ','.join(f"{key}:{value}" for key, value in self.bounding_box.items())

        entities_element = ET.SubElement(element, 'Entities')
        for entity in self.entities:
            entities_element.append(entity.to_xml())

        speech_bubbles_element = ET.SubElement(element, 'SpeechBubbles')
        for speech_bubble in self.speech_bubbles:
            speech_bubbles_element.append(speech_bubble.to_xml())

        # Return the XML element
        return element