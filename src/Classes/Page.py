from enum import Enum
from typing import List
from .Panel import Panel

class PageType(Enum):
    INDEX = 1
    CHAPTER_HEAD = 2
    SINGLE = 3
    DUAL = 4
    TRIVIA = 5

class Page:

    def __init__(self,page_image:List[List[int]],page_index: int,page_type: PageType,height: float, width: float,panels:List[Panel]):
        self.page_image = page_image
        self.page_index = page_index
        self.page_type = page_type
        self.height = height
        self.width = width
        self.panels = panels if panels else []
