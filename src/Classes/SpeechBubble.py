from enum import Enum
from typing import Tuple,Optional,List
from .Entity import Entity

class SpeechBubbleType(Enum):
    NARRATOR = 1
    SPEECH = 2
    THOUGHTS = 3
    EFFECT = 4
    SHOUT = 5

class SpeechBubble:
    speaker: Optional['List[Entity]'] = None
    image = None
    
    def __init__ (self,type: SpeechBubbleType, text: str, bounding_box: Tuple[float,float,float,float],image):
        self.type = type
        self.text = text
        self.bounding_box = bounding_box
        self.image = image

    def set_speaker(self, speaker: List[Entity]):
        self.speaker = speaker