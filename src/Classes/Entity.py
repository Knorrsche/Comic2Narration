from .EntityTemplate import EntityTemplate
from typing import Tuple

class Entity:

    entity_template: EntityTemplate = None

    def __init__ (self,bounding_box: Tuple[float,float,float,float]):
        self.bounding_box = bounding_box
    