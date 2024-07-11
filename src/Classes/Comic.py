from .Series import Series
from .Page import Page
from typing import Optional, List
import xml.etree.ElementTree as ET 

class Comic:
    def __init__(self, name: str, volume: int, main_series: Series, secondary_series: Optional['List[Series]'],page_pairs: List[tuple[Optional[Page],Optional[Page]]]):
        self.name = name
        self.volume = volume
        self.main_series = main_series
        self.secondary_series = secondary_series
        self.page_pairs = page_pairs

    #TODO: create series object
    def to_xml(self):
        element = ET.Element('Comic')
        ET.SubElement(element, 'Name').text = self.name
        ET.SubElement(element, 'Volume').text = str(self.volume)
        #ET.SubElement(element, 'MainSeries').text = self.main_series.name
        ET.SubElement(element, 'MainSeries').text = 'MainSeries'
        
        secondary_series_element = ET.SubElement(element, 'SecondarySeries')
        for series in self.secondary_series:
            series_name_element = ET.SubElement(secondary_series_element, 'Series')
            #ET.SubElement(series_name_element, 'Name').text = series.name
            ET.SubElement(series_name_element, 'Name').text = 'SecondarySeries'

        page_pairs_element = ET.SubElement(element, 'PagePairs')
        for pair in self.page_pairs:
            pair_element = ET.SubElement(page_pairs_element, 'PagePair')
            if pair[0] and pair[0] is not None:
                pair_element_left = ET.SubElement(pair_element, 'LeftPage')
                pair_element_left.append(pair[0].to_xml())
            if pair[1]and pair[1] is not None:
                pair_element_right = ET.SubElement(pair_element, 'RightPage')
                pair_element_right.append(pair[1].to_xml())

        return element