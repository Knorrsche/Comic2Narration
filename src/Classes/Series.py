from .EntityTemplate import EntityTemplate
from typing import List

class Series:
    entity_templates: EntityTemplate = None
   
    def __init__(self, name: str):
        self.name = name

    def set_entity_templates(self, entity_templates: List[EntityTemplate]):
        self.entity_templates = entity_templates
