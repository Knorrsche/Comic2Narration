import logging
import threading
from typing import Optional, List
from Classes.Panel import Panel
from Classes.Comic import Comic
from Classes.Page import Page, PageType
from Classes.Series import Series
from Utils import ImageUtils as iu
from inference_sdk import InferenceHTTPClient
import ollama
from PIL import Image
import io
import cv2

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class ComicPreprocessor:
    current_comic: Comic
    export_path: str

    def __init__(self, name: str, volume: int, main_series: Series, secondary_series: Optional[List[Series]],
                 rgb_arrays):
        page_pairs = self.convert_array_to_page_pairs(rgb_arrays)
        self.current_comic = self.convert_images_to_comic(name, volume, main_series, secondary_series, page_pairs)
        self.detect_panels()

    @staticmethod
    def convert_images_to_comic(name: str, volume: int, main_series: Series,
                                secondary_series: Optional[List[Series]], page_pairs):
        return Comic(name, volume, main_series, secondary_series, page_pairs)

    # TODO: refactor loop and use counter
    def convert_array_to_page_pairs(self, rgb_arrays):
        page_pairs = []
        i = 0
        while i < len(rgb_arrays):
            rgb_array = rgb_arrays[i]
            height, width, _ = rgb_array.shape

            if i == 0:
                first_page = None
                second_page = Page(page_image=rgb_array, page_index=i, page_type=self.classify_page(), height=height, width=width)
            else:
                first_page = Page(page_image=rgb_array, page_index=i, page_type=self.classify_page(), height=height, width=width)

                if i + 1 < len(rgb_arrays):
                    rgb_array_next = rgb_arrays[i + 1]
                    height_next, width_next, _ = rgb_array_next.shape
                    second_page = Page(page_image=rgb_array_next, page_index=i + 1, page_type=self.classify_page(), height=height_next, width=width_next)
                    i += 1
                else:
                    second_page = None

            page_pairs.append((first_page, second_page))
            i += 1

        return page_pairs

    # TODO: Create Page Classifier
    @staticmethod
    def classify_page():
        return PageType.SINGLE

    # TODO: Image Descriptor
    def describe_image(self,image):

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        image_bytes = io.BytesIO()
        pil_image.save(image_bytes, format='PNG')
        image_bytes = image_bytes.getvalue()

        message = {
            'role': 'user',
            'content': 'Only answer with what is happening and answer in a short sentence. What is happening in the comic panel?',
            'images': [image_bytes]
        }

        res = ollama.chat(
            model='llava',
            messages=[message]
        )

        description = res['message']['content']
        return description

    # TODO: Refactor with extract_speech_bubbles
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

    def handle_detect_panels(self, page: Page):

        CLIENT_PANEL = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key="bbQfI1dqQMBQJJnHI4AU"
        )

        result_panels = CLIENT_PANEL.infer(page.page_image, model_id="comic-panel-detectors/7")
        panels = []

        for panel_data in result_panels['predictions']:
            if panel_data['confidence'] < 0.3:
                continue

            panel_image = iu.image_from_bbox(page.page_image, panel_data)

            description = self.describe_image(panel_image)
            panel = Panel(description, panel_data, panel_image)
            panels.append(panel)

        # TODO: Sort Panels by position
        panels = sorted(panels, key=lambda p: ( (p.bounding_box['y']-p.bounding_box['height']/2),
                                                (p.bounding_box['x'])-p.bounding_box['width']/2))
        page.panels = panels
