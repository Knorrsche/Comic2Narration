from typing import Tuple,List
from .Entity import Entity
from .SpeechBubble import SpeechBubble


class Panel:
    def __init__(self,description: str, bounding_box: Tuple[float,float,float,float],image):
        self.description = description
        self.bounding_box = bounding_box
        self.image = image
        self.entities: List[Entity] = []
        self.speech_bubbles: List[SpeechBubble]= []
