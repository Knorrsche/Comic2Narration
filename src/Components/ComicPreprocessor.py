import logging
import threading
from typing import Optional, List
from src.Classes.Panel import Panel
from src.Classes.Comic import Comic
from src.Classes.Page import Page, PageType
from src.Classes.Series import Series
from src.Utils import ImageUtils as iu
from inference_sdk import InferenceHTTPClient

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
            if panel_data['confidence'] < 0.3:
                continue

            panel_image = iu.image_from_bbox(page.page_image,panel_data)
   
            description = self.describe_image()
            panel = Panel(description, panel_data,panel_image)
            panels.append(panel)
            
        # TODO: Sort Panels by position
        page.panels = panels
    
    
