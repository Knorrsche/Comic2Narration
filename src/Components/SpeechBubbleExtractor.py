from Classes.Comic import Comic
from Classes.SpeechBubble import SpeechBubble,SpeechBubbleType
from Classes.Page import Page
from PIL import Image
from inference_sdk import InferenceHTTPClient
import numpy as np
import threading
import logging
from Utils import ImageUtils as iu

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class SpeechBubbleExtractor:


    def __init__ (self,current_comic:Comic):
        self.current_comic = current_comic
        self.extract_speech_bubbles()


    #TODO: Refactor with detect_panel
    def extract_speech_bubbles(self):

        for i, page_pair in enumerate(self.current_comic.page_pairs):
            threads = [] 
            for page in page_pair:
                if not page:
                    continue
                thread = threading.Thread(target=self.handle_extract_speech_bubbles,args=(page,))
                threads.append(thread)
                logging.debug(f'Starting Speechbubble Extraction thread for page {page.page_index}')
                thread.start()

            for thread in threads:
                thread.join()
                logging.debug(f'Thread for Speechbubble Extraction of page_pair {i} finished')   
            logging.debug('\n')


    #TODO: extract from the whole page instead of just each panel
    def handle_extract_speech_bubbles(self,page:Page):
        
        CLIENT_BUBBLE = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key="bbQfI1dqQMBQJJnHI4AU"
        )

        page_image = page.page_image 

        result_bubbles = CLIENT_BUBBLE.infer(page_image, model_id="bubble-detection-gbjon/2")
                
        for speech_bubble_data in result_bubbles['predictions']:
            if speech_bubble_data['confidence'] < 0.3:
                continue
            description = self.ocr_text()
            speech_bubble_type:SpeechBubbleType = self.classify_speech_bubble()
            speech_bubble_image = iu.image_from_bbox(page_image,speech_bubble_data)
            speech_bubble = SpeechBubble(speech_bubble_type,description,speech_bubble_data,speech_bubble_image)
            
            for panel in page.panels:
                if iu.is_bbox_overlapping(speech_bubble.bounding_box,panel.bounding_box,):
                    panel.speech_bubbles.append(speech_bubble)
        

    # TODO: Train a Comic OCR
    def ocr_text(self):
        return ''
    
    # TODO: Train a classifier
    def classify_speech_bubble(self):
        return SpeechBubbleType.SPEECH