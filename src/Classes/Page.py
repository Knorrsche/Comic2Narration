from enum import Enum
from typing import List,Optional
from .Panel import Panel
from Utils import ImageUtils as iU
import xml.etree.ElementTree as eT


class PageType(Enum):
    INDEX = 1
    CHAPTER_HEAD = 2
    SINGLE = 3
    DUAL = 4
    TRIVIA = 5


class Page:
    bbox_color_panel = (0, 255, 0)
    bbox_thickness_panel = 2
    bbox_color_speech_bubble = (255, 0, 0)
    bbox_thickness_speech_bubble = 2

    # TODO: refactor height and width ou
    def __init__(self, page_index: int, page_type: PageType,
                     panels: Optional[List[Panel]] = None,
                     page_image= None,
                     height: Optional[float] = None, width: Optional[float] = None):
            self.page_index = page_index
            self.page_type = page_type
            self.panels = panels if panels else []
            self.page_image = page_image
            self.height = height
            self.width = width

    def annotated_image(self, draw_panels: bool, draw_speech_bubbles: bool):
        new_image = self.page_image.copy()

        for panel in self.panels:
            new_image = iU.draw_bounding_box(
                new_image, panel.bounding_box, self.bbox_color_panel, self.bbox_thickness_panel
            ) if draw_panels is True else new_image

            if not draw_speech_bubbles:
                continue

            for speech_bubble in panel.speech_bubbles:
                new_image = iU.draw_bounding_box(
                    new_image, speech_bubble.bounding_box, self.bbox_color_speech_bubble,
                    self.bbox_thickness_speech_bubble
                )

        return new_image

    def to_xml(self):
        element = eT.Element('Page')
        eT.SubElement(element, 'Index').text = str(self.page_index)
        eT.SubElement(element, 'Type').text = self.page_type.name

        panel_element = eT.SubElement(element, 'Panels')
        for panel in self.panels:
            panel_element.append(panel.to_xml())

        return element
