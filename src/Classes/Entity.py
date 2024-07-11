from .EntityTemplate import EntityTemplate
from typing import Tuple
import xml.etree.ElementTree as ET

class Entity:

    #entity_template: EntityTemplate = None

    def __init__ (self,bounding_box: Tuple[float,float,float,float]):
        self.bounding_box = bounding_box
        
    def to_xml(self):
        element = ET.Element('Entity')
        ET.SubElement(element, 'Name').text = self.get_entity_template_name
        bbox = ET.SubElement(element, 'BoundingBox')
        bbox.text = ','.join(map(str, self.bounding_box))
        return element

    #TODO: Update if entity & entity_template are getting implemented
    def get_entity_template_name(self):
        #return self.entity_template.name      
        return 'Character'