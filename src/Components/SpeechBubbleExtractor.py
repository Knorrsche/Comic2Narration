from collections import Counter

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
import spacy
import comiq

from src.Components import ComicPreprocessor

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class SpeechBubbleExtractor:

    def __init__(self, current_comic: Comic, comic_preprocessor: ComicPreprocessor):
        self.current_comic = current_comic
        #self.extract_speech_bubbles()
        #comic_preprocessor.guess_speakers()

    #TODO: Refactor with detect_panel
    def extract_speech_bubbles(self):
        comiq.set_api_key("AIzaSyCoUfTjZU-zNZ2lNKY_BnDuyNTu8lHQ9EM")
        for i, page_pair in enumerate(self.current_comic.page_pairs):
            for page in page_pair:
                if not page:
                    continue
                print(f'Starting Speechbubble Extraction thread for page {page.page_index}')
                self.handle_extract_speech_bubbles(page)

    def handle_extract_speech_bubbles(self, page: Page):
        nlp = spacy.load("en_core_web_trf")

        page_image = page.page_image
        if page_image is None:
            return
        result_bubbles = comiq.extract(page_image)

        old_speech_bubble_list = [[]]
        for panel in page.panels:
            old_speech_bubble_list.append(panel.speech_bubbles)
            panel.speech_bubbles = []

        for result_bubble in result_bubbles:
            speech_bubble_type: SpeechBubbleType = result_bubble['type']
            box_coordinates = result_bubble['text_box']
            bounding_box = {
                "x": box_coordinates[1],
                "y": box_coordinates[0],
                "width": box_coordinates[3] - box_coordinates[1],
                "height": box_coordinates[2] - box_coordinates[0],
            }
            speech_bubble_image = iu.image_from_bbox(page_image, bounding_box)
            description = result_bubble['text']
            if not description is None:
                person_list = self.apply_named_entitiy_recognition(description, nlp)
            if description is None or description == "":
                continue
            speech_bubble = SpeechBubble(speech_bubble_type, description, bounding_box, speech_bubble_image)

            for i,panel in enumerate(page.panels):
                old_speech_bubbles = old_speech_bubble_list[i]
                old_bubble = next(
                    (bubble for bubble in old_speech_bubbles if
                     iu.calculate_iou(bubble.bounding_box, speech_bubble.bounding_box) > 0.3),
                    None
                )
                if old_bubble:
                    speech_bubble.speaker_id = old_bubble.speaker_id
                    if old_bubble.trail:
                        speech_bubble.trail = old_bubble.trail
                    break

            speech_bubble.person_list = person_list

            for panel in page.panels:
                if iu.is_bbox_overlapping(panel.bounding_box, speech_bubble.bounding_box):
                    panel.speech_bubbles.append(speech_bubble)

        for panel in page.panels:
            speech_bubbles = panel.speech_bubbles
            panel.speech_bubbles = sorted(speech_bubbles,
                                          key=lambda p: ((p.bounding_box['y'] - p.bounding_box['height']),
                                                         (p.bounding_box['x']) - p.bounding_box['width']))

    def apply_named_entitiy_recognition(self, text, nlp):

        doc = nlp(text)

        person_entities = [(ent.text, ent.start, ent.end) for ent in doc.ents if ent.label_ == "PERSON"]
        print(person_entities)

        def merge_names(entities):
            merged_entities = []
            i = 0
            while i < len(entities):
                if i + 1 < len(entities) and entities[i][2] == entities[i + 1][1] - 1:
                    merged_name = entities[i][0] + " " + entities[i + 1][0]
                    merged_entities.append(merged_name)
                    i += 2
                else:
                    merged_entities.append(entities[i][0])
                    i += 1
            return merged_entities

        names = merge_names(person_entities)
        return names
