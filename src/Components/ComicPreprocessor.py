import base64
import logging
import os
import threading
from typing import Optional, List
from Classes.Panel import Panel
from Classes.Comic import Comic
from Classes.Page import Page, PageType
from Classes.Series import Series
from Classes.Entity import Entity
from Utils import ImageUtils as iu
from inference_sdk import InferenceHTTPClient
from gradio_client import Client, handle_file
import ollama
import tempfile
from PIL import Image
import io
import cv2
import numpy as np

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class ComicPreprocessor:
    current_comic: Comic
    export_path: str

    str_general_instuction = ('GENERAL INSTRUCTIONS:\n'
                              '    You are a expert in Comic summarization. Your task is to describe what is happening in an image of a comic panel.\n'
                              '    You will also get context in form of text from speech bubbles, but you are not allowed to return the words from the speech bubbles.\n'
                              '    You should only use the text as a reference to improve the panel description.\n'
                              '    Answer in a short sentence and try to only describe information\'s that are needed to understand the narrative. \n'
                              '    Do not talk about speech bubbles and do not recite them.\n'
                              '    When answering do not mention that this is an image or comic. Answer like you would write a Novel.\n'
                              '    \n'
                              '    Example: A shadow is fighting against a Robot at night. A police officer is helping.\n'
                              '    \n'
                              )

    def __init__(self, name: str, volume: int, main_series: Series, secondary_series: Optional[List[Series]],
                 rgb_arrays):
        page_pairs = self.convert_array_to_page_pairs(rgb_arrays)
        self.current_comic = self.convert_images_to_comic(name, volume, main_series, secondary_series, page_pairs)
        self.detect_panels()
        self.detect_entities()
        self.create_tags()

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
                second_page = Page(page_image=rgb_array, page_index=i, page_type=self.classify_page(), height=height,
                                   width=width)
            else:
                first_page = Page(page_image=rgb_array, page_index=i, page_type=self.classify_page(), height=height,
                                  width=width)

                if i + 1 < len(rgb_arrays):
                    rgb_array_next = rgb_arrays[i + 1]
                    height_next, width_next, _ = rgb_array_next.shape
                    second_page = Page(page_image=rgb_array_next, page_index=i + 1, page_type=self.classify_page(),
                                       height=height_next, width=width_next)
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
    def describe_image(self, image):

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        image_bytes = io.BytesIO()
        pil_image.save(image_bytes, format='PNG')
        image_bytes = image_bytes.getvalue()

        message = {
            'role': 'user',
            'content': self.str_general_instuction,
            'images': [image_bytes]
        }

        res = ollama.chat(
            model='llava',
            messages=[message]
        )

        description = res['message']['content']
        return description

    def improve_panel_descriptions(self):
        threads = []

        for i, page_pair in enumerate(self.current_comic.page_pairs):
            for page in page_pair:
                if not page:
                    continue

                thread = threading.Thread(target=self.handle_improve_panels_neighbors, args=(page,))
                threads.append(thread)
                logging.debug(f'Starting Description Improvement with neighbors thread for page {page.page_index}')
                thread.start()

            for thread in threads:
                thread.join()
                logging.debug(f'Threads for Panel Extraction with neighbors of page_pair {i} finished')
            threads = []
            logging.debug('\n')

        for i, page_pair in enumerate(self.current_comic.page_pairs):
            for page in page_pair:
                if not page:
                    continue

                thread = threading.Thread(target=self.handle_improve_panels_page, args=(page,))
                threads.append(thread)
                logging.debug(f'Starting Description Improvement with neighbors thread for page {page.page_index}')
                thread.start()

            for thread in threads:
                thread.join()
                logging.debug(f'Threads for Panel Extraction with neighbors of page_pair {i} finished')
            threads = []
            logging.debug('\n')

    def handle_improve_panels_page(self, page):
        panels = page.panels
        str_page_context = 'This is the context of the whole Page. Use this to improve the description of the Current Panel. \n'
        # Make page get transcript function
        for i, panel in enumerate(panels):
            str_page_context += f'Panel {i + 1} Description: {panel.description}\n'
            #str_page_context += f'Panel {i + 1} Speech Bubbles: {panel.get_transcript}\n'

        for i, panel in enumerate(panels):
            str_context = ''
            str_context += str_page_context
            str_context += f'Current Panel {i + 1} Description: {panel.description}\n'
            #str_context += f'Current Panel {i+1} Speech Bubbles: {panel.get_transcript}\n'

            image_rgb = cv2.cvtColor(panel.image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)

            image_bytes = io.BytesIO()
            pil_image.save(image_bytes, format='PNG')
            image_bytes = image_bytes.getvalue()

            message = {
                'role': 'user',
                'content': str_context + '\n' + '\n' + self.str_general_instuction + '\n' + '\n' + 'Improve the Current Panel description following the general istruction and use as few words as possible',
                'images': [image_bytes]
            }

            res = ollama.chat(
                model='llava',
                messages=[message]
            )

            description = res['message']['content']
            panel.descriptions.append(description)
            panel.description = description

    def handle_improve_panels_neighbors(self, page):
        panels = page.panels
        for i, panel in enumerate(panels):
            str_context = ''
            current_panel = panel
            if i != 0:
                previous_panel = panels[i - 1]
                str_context += f'Previous Panel Description: {previous_panel.description}\n'
                #str_context += f'Previous Panel Speech Bubbles: {previous_panel.get_transcript}\n'
            if i != len(panels) - 1:
                next_panel = panels[i + 1]
                str_context += f'Next Panel Description: {next_panel.description}\n'
                #str_context += f'Next Panel Speech Bubbles: {next_panel.get_transcript}\n'
            str_context += f'Current Panel Description: {current_panel.description}\n'
            #str_context += f'Current Panel Speech Bubbles: {current_panel.get_transcript}\n'

            image_rgb = cv2.cvtColor(current_panel.image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)

            image_bytes = io.BytesIO()
            pil_image.save(image_bytes, format='PNG')
            image_bytes = image_bytes.getvalue()

            message = {
                'role': 'user',
                'content': str_context + '\n' + '\n' + self.str_general_instuction + '\n' + '\n' + 'Improve the Current Panel description following the general istruction and use as few words as possible.',
                'images': [image_bytes]
            }

            res = ollama.chat(
                model='llava',
                messages=[message]
            )

            description = res['message']['content']
            panel.descriptions.append(description)
            panel.description = description

    # TODO find way for threading
    def create_tags(self):
        threads = []

        for i, page_pair in enumerate(self.current_comic.page_pairs):
            for page in page_pair:
                if not page:
                    continue

                logging.debug(f'Starting Tag creation thread for page {page.page_index}')
                self.handle_create_tags(page)
            logging.debug('\n')

    def handle_create_tags(self, page):
        client = Client("SmilingWolf/wd-tagger")

        for panel in page.panels:
            for entity in panel.entities:
                pil_image = Image.fromarray(entity.image)

                pil_image.save('tmp.jpg')
                image_input = handle_file('tmp.jpg')

                result = client.predict(
                        image=image_input,
                        model_repo="SmilingWolf/wd-swinv2-tagger-v3",
                        general_thresh=0.5,
                        general_mcut_enabled=False,
                        character_thresh=0.85,
                        character_mcut_enabled=False,
                        api_name="/predict"
                    )

                confidences = result[3]['confidences']
                tag_confidence_tuples = [(item['label'], item['confidence']) for item in confidences]

                entity.tags = tag_confidence_tuples

                # TODO: add to .env as name
                os.remove('tmp.jpg')


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
            panel.descriptions.append(panel.description)
            panels.append(panel)

        panels = sorted(panels, key=lambda p: ((p.bounding_box['y'] - p.bounding_box['height'] / 2),
                                               (p.bounding_box['x']) - p.bounding_box['width'] / 2))
        page.panels = panels

    def detect_entities(self):
        threads = []

        for i, page_pair in enumerate(self.current_comic.page_pairs):
            for page in page_pair:
                if not page:
                    continue

                thread = threading.Thread(target=self.handel_detect_entity, args=(page,))
                threads.append(thread)
                logging.debug(f'Starting Entity Extraction thread for page {page.page_index}')
                thread.start()

            for thread in threads:
                thread.join()
                logging.debug(f'Threads for Entity Extraction of page_pair {i} finished')
            threads = []
            logging.debug('\n')

    def handel_detect_entity(self, page: Page):

        CLIENT = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key="bbQfI1dqQMBQJJnHI4AU"
        )

        result_entities = CLIENT.infer(page.page_image, model_id="marvel_chars/1")
        entities = []

        for entity_data in result_entities['predictions']:
            if entity_data['confidence'] < 0:
                continue

            entity = Entity(entity_data)
            entity.image = iu.image_from_bbox(page.page_image, entity_data)
            entities.append(entity)

            for panel in page.panels:
                if iu.is_bbox_overlapping(panel.bounding_box, entity.bounding_box):
                    panel.entities.append(entity)

        for panel in page.panels:
            entities = panel.entities
            panel.entities = sorted(entities,
                                    key=lambda p: ((p.bounding_box['y'] - p.bounding_box['height'] / 2),
                                                   (p.bounding_box['x']) - p.bounding_box['width'] / 2))


