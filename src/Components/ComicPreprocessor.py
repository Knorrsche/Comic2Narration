from Classes.Comic import Comic
from Classes.Series import Series
from Classes.Page import Page,PageType
from Classes.Panel import Panel
from typing import Optional, List
from .ComicPdfReader import convert_pdf_to_image
from inference_sdk import InferenceHTTPClient
from PIL import Image
import numpy as np



class ComicPreprocessor:
    current_comic: Comic
    export_path: str

    def __init__(self, rgb_arrays, name: str, volume: int, main_series: Series, secondary_series: Optional[List[Series]]):
        page_images = self.convert_array_to_pages(self,rgb_arrays)
        self.current_comic = self.convert_images_to_comic(page_images, name, volume, main_series, secondary_series)
        self.detect_panels(self)

    @staticmethod
    def convert_images_to_comic(rgb_arrays, name: str, volume: int, main_series: Series, secondary_series: Optional[List[Series]]):
        return Comic(name, volume, main_series, secondary_series, rgb_arrays)

    @staticmethod
    def convert_array_to_pages(self,rgb_arrays):
        pages = []
        for i in range(len(rgb_arrays)):
      
            rgb_array = rgb_arrays[i]
            height, width, _ = rgb_array.shape
            pages.append(Page(rgb_array,i,self.classify_page(),height,width,[]))
        
        return pages
    
    # TODO: Create Page Classifier
    def classify_page(self):
        return PageType.SINGLE
    
    # TODO: Image Descriptior
    def describe_image(self):
        return ''

    # TODO: DRY!!! Refactor code -> Used in SpeechBubbleExtractor and ComicDisplayPage
    def load_panel_image_from_page(self, page, bbox):
        x = int(bbox['x'] - bbox['width'] / 2)
        y = int(bbox['y'] - bbox['height'] / 2)
        w = int(bbox['width'])
        h = int(bbox['height'])

        panel_image = page.page_image[y:y+h, x:x+w, :]
        return panel_image


    @staticmethod
    def detect_panels(self):
        CLIENT_PANEL = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key="bbQfI1dqQMBQJJnHI4AU"
        )

        for page in self.current_comic.pages:
            result_panels = CLIENT_PANEL.infer(page.page_image, model_id="comic-panel-detectors/7")
            panels = []
            
            for panel_data in result_panels['predictions']:
    
                panel_image = self.load_panel_image_from_page(page,panel_data)
   
                description = self.describe_image()
                panel = Panel(description, panel_data,panel_image)
                panels.append(panel)
            
            # TODO: Sort Panels by position
            self.link_panels(panels)
            page.panels = panels
    
    def link_panels(self, panels):
        for i in range(len(panels)):
            prev_panel = panels[i-1] if i > 0 else None
            next_panel = panels[i+1] if i < len(panels)-1 else None
            panels[i].set_panels(next_panel, prev_panel)