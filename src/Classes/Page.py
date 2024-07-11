from enum import Enum
from typing import List
from .Panel import Panel
from src.Utils import ImageUtils as iu

class PageType(Enum):
    INDEX = 1
    CHAPTER_HEAD = 2
    SINGLE = 3
    DUAL = 4
    TRIVIA = 5

class Page:

    bbox_color_panel = (0, 255, 0)
    bbox_thickness_panel = 2
    bbox_color_speech_bubble = (255,0, 0)
    bbox_thickness_speech_bubble = 2


    def __init__(self,page_image:List[List[int]],page_index: int,page_type: PageType,height: float, width: float,panels:List[Panel]):
        self.page_image = page_image
        self.page_index = page_index
        self.page_type = page_type
        self.height = height
        self.width = width
        self.panels = panels if panels else []


    def annotateted_image(self):
        new_image = self.page_image.copy()

        for panel in self.panels:
            new_image = iu.draw_bounding_box(
                new_image, panel.bounding_box, self.bbox_color_panel, self.bbox_thickness_panel
            )

            for speech_bubble in panel.speech_bubbles:
                new_image = iu.draw_bounding_box(
                new_image, speech_bubble.bounding_box, self.bbox_color_speech_bubble, self.bbox_thickness_speech_bubble
                )

        return new_image

