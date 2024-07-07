from Classes.Comic import Comic
from Classes.Series import Series
from Classes.Page import Page,PageType
from Classes.Panel import Panel
from typing import Optional, List
from .ComicPdfReader import convert_pdf_to_image
from inference_sdk import InferenceHTTPClient
import threading
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class ComicPreprocessor:
    current_comic: Comic
    export_path: str

    def __init__(self, name: str, volume: int, main_series: Series, secondary_series: Optional[List[Series]],rgb_arrays):
        page_pairs = self.convert_array_to_page_pairs(rgb_arrays)
        self.current_comic = self.convert_images_to_comic(name, volume, main_series, secondary_series,page_pairs)
        self.detect_panels()

    def convert_images_to_comic(self,name: str, volume: int, main_series: Series, secondary_series: Optional[List[Series]],page_pairs):
        return Comic(name, volume, main_series, secondary_series, page_pairs)

    def convert_array_to_page_pairs(self,rgb_arrays):
        page_pairs = []
        i = 0
        while i < len(rgb_arrays):
            rgb_array = rgb_arrays[i]
            height, width, _ = rgb_array.shape
            
            if i == 0:
                first_page = None
                second_page = Page(rgb_array, i, self.classify_page(), height, width, [])
            else:
                first_page = Page(rgb_array, i, self.classify_page(), height, width, [])
                
                if i + 1 < len(rgb_arrays):
                    rgb_array_next = rgb_arrays[i + 1]
                    height_next, width_next, _ = rgb_array_next.shape
                    second_page = Page(rgb_array_next, i + 1, self.classify_page(), height_next, width_next, [])
                    i+=1
                else:
                    second_page = None
   
            page_pairs.append((first_page, second_page))
            i +=1
    
        return page_pairs
    
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


    #TODO: Refactor with extract_speech_bubbles
    def detect_panels(self):
        threads = []

        for i, page_pair in enumerate(self.current_comic.page_pairs):
            for page in page_pair:
                if not page:
                    continue
                
                thread = threading.Thread(target=self.handle_detect_panels, args=(page,))
                threads.append(thread)
                logging.debug(f'Starting Panel Extraction thread for page {page.page_index}')
                thread.start()

            for thread in threads:
                thread.join()
                logging.debug(f'Threads for Panel Extraction of page_pair {i} finished')
            threads = []
            logging.debug('\n')

    def handle_detect_panels(self,page:Page):
           
        CLIENT_PANEL = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key="bbQfI1dqQMBQJJnHI4AU"
        )

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
    
