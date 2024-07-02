from Classes.Comic import Comic
from Classes.SpeechBubble import SpeechBubble,SpeechBubbleType
from PIL import Image
from inference_sdk import InferenceHTTPClient
import numpy as np

class SpeechBubbleExtractor:


    def __init__ (self,current_comic:Comic):
        self.current_comic = current_comic
        self.extract_speech_bubbles()

    def image_from_bbox(self, panel, bbox):
        x = int(bbox['x'] - bbox['width'] / 2)
        y = int(bbox['y'] - bbox['height'] / 2)
        w = int(bbox['width'])
        h = int(bbox['height'])

        speech_bubble_image = panel.image[y:y+h, x:x+w, :]
        return speech_bubble_image


    def extract_speech_bubbles(self):
        CLIENT_BUBBLE = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key="bbQfI1dqQMBQJJnHI4AU"
        )

        pages = self.current_comic.pages

        for i in range(len(pages)):
            page = pages[i]
            for p in page.panels:

                speech_bubbles = []
                
                page_image = p.image
                
                result_bubbles = CLIENT_BUBBLE.infer(page_image, model_id="bubble-detection-gbjon/2")
                
                for speech_bubble_data in result_bubbles['predictions']:
                    description = self.ocr_text()
                    speech_bubble_type = self.classify_speech_bubble()
                    speech_bubble_image = self.image_from_bbox(p,speech_bubble_data)
                    speech_bubble = SpeechBubble(description, speech_bubble_type,speech_bubble_data,speech_bubble_image)
                    speech_bubbles.append(speech_bubble)
           
            p.speech_bubbles = speech_bubbles


    # TODO: Train a Comic OCR
    def ocr_text(self):
        return ''
    
    # TODO: Train a classifier
    def classify_speech_bubble(self):
        return SpeechBubbleType.SPEECH