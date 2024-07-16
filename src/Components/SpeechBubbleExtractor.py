from Classes.Comic import Comic
from Classes.SpeechBubble import SpeechBubble, SpeechBubbleType
from Classes.Page import Page
from PIL import Image
from inference_sdk import InferenceHTTPClient
import numpy as np
import threading
import logging
from Utils import ImageUtils as iu
import pytesseract
import xml.sax.saxutils as saxutils

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class SpeechBubbleExtractor:

    def __init__(self, current_comic: Comic):
        self.current_comic = current_comic
        self.extract_speech_bubbles()

    #TODO: Refactor with detect_panel
    def extract_speech_bubbles(self):

        for i, page_pair in enumerate(self.current_comic.page_pairs):
            threads = []
            for page in page_pair:
                if not page:
                    continue
                thread = threading.Thread(target=self.handle_extract_speech_bubbles, args=(page,))
                threads.append(thread)
                logging.debug(f'Starting Speechbubble Extraction thread for page {page.page_index}')
                thread.start()

            for thread in threads:
                thread.join()
                logging.debug(f'Thread for Speechbubble Extraction of page_pair {i} finished')
            logging.debug('\n')

    #TODO: extract from the whole page instead of just each panel
    def handle_extract_speech_bubbles(self, page: Page):

        CLIENT_BUBBLE = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key="bbQfI1dqQMBQJJnHI4AU"
        )

        page_image = page.page_image

        result_bubbles = CLIENT_BUBBLE.infer(page_image, model_id="bubble-detection-gbjon/2")
        for speech_bubble_data in result_bubbles['predictions']:
            if speech_bubble_data['confidence'] < 0.3:
                continue
            speech_bubble_type: SpeechBubbleType = self.classify_speech_bubble()
            speech_bubble_image = iu.image_from_bbox(page_image, speech_bubble_data)
            description = self.ocr_text(speech_bubble_image)
            speech_bubble = SpeechBubble(speech_bubble_type, description, speech_bubble_data, speech_bubble_image)

            for panel in page.panels:
                if iu.is_bbox_overlapping(panel.bounding_box,speech_bubble.bounding_box):
                    panel.speech_bubbles.append(speech_bubble)

        for panel in page.panels:
            speech_bubbles = panel.speech_bubbles
            panel.speech_bubbles = sorted(speech_bubbles, key=lambda p: ((p.bounding_box['y'] - p.bounding_box['height'] / 2),
                                                   (p.bounding_box['x']) - p.bounding_box['width'] / 2))
    # TODO: Train a Comic OCR
    def ocr_text(self, image):
        # TODO: add tesseract path to .env
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        #pytesseract.pytesseract.tesseract_cmd=r'/usr/bin/tesseract'
        text = pytesseract.image_to_string(image)
        text = bytes(text, 'utf-8').decode('utf-8', 'ignore')
        return text

    # TODO: Train a classifier
    def classify_speech_bubble(self):
        return SpeechBubbleType.SPEECH
