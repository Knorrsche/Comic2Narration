from typing import Tuple,Optional,List
from .Entity import Entity
from .SpeechBubble import SpeechBubble
from PIL import Image
import numpy as np

class Panel:
    next_panel: Optional['Panel'] = None
    prev_panel: Optional['Panel'] = None
    entities: Optional['List[Entity]'] = None
    speech_bubbles: Optional['List[SpeechBubble]'] = []
    bounding_box: Tuple[float,float,float,float] = None
    image = None
    
    def __init__(self,description: str, bounding_box: Tuple[float,float,float,float],image):
        self.description = description
        self.bounding_box = bounding_box
        self.image = image

    def set_panels(self,next_panel: Optional['Panel'], prev_panel: Optional['Panel']):
        self.next_panel = next_panel
        self.prev_panel = prev_panel

    def set_entities(self,entities: List['Entity']):
        self.entities = entities
    
    def set_speech_bubbles(self, speech_bubbles: List['SpeechBubble']):
        self.speech_bubbles = speech_bubbles

    def get_next_panel(self) -> Optional['Panel']:
        return self.next_panel

    def get_prev_panel(self) -> Optional['Panel']:
        return self.prev_panel