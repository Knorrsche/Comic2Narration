import base64
import json
import logging
import os
import random
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
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.morphology import dilation
from scipy import ndimage as ndi
from skimage.measure import label
from skimage.color import label2rgb
from skimage.measure import regionprops
import imageio
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from transformers import AutoModel
import torch
from src.entity_detector import wdv3_timm
from src.Classes import SpeechBubble, SpeechBubbleType
import google.generativeai as genai

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class ComicPreprocessor:
    current_comic: Comic
    export_path: str
    entity_id_counter = 1
    comic_summarization_prompt = (
        "GENERAL INSTRUCTIONS:\n"
        "You are an expert in comic panel summarization. Your task is to analyze the comic panels and provide a comprehensive, continuous narrative by combining individual panels into larger narrative arcs. These arcs should progress linearly without any gaps or interruptions, forming a coherent story timeline. You will be given previous narrative summaries and your job is to expand and refine them incrementally based on new panels or pages.\n\n"

        "Your goal is to create a seamless flow of events, where narrative arcs do not overlap or create disjointed stories. Instead, arcs should be aligned sequentially, following one another to build a continuous timeline.\n\n"

        "Key Guidelines:\n"
        "1. **Focus on Continuous Narrative Arcs**:\n"
        "   - Narrative arcs must occur in a continuous line, meaning no gaps or holes should exist within an arc. If an arc spans multiple panels, the events should follow one another smoothly without skipping over unrelated content.\n"
        "   - Arcs should align sequentially and flow into each other logically. One arc cannot span into the middle of another; instead, arcs should build upon one another in a chronological, cohesive timeline.\n"
        "   - **Example**: A fight scene unfolding over multiple panels must be described as one continuous arc. You cannot describe a scene from one arc, skip several panels, and return to it later without finishing the previous one.\n\n"

        "2. **Avoid Overlapping Arcs**:\n"
        "   - Arcs must not overlap. Once an arc concludes, the next arc should start from the following panel in a clear and distinct way. Ensure that there is no overlap where two arcs are describing the same moment from different perspectives.\n"
        "   - **Example**: If Arc 1 describes a chase scene, Arc 2 should follow directly after, showing the next stage of the pursuit or a new development rather than splitting the chase across two separate arcs.\n\n"

        "3. **Infer from Dialogue, But Do Not Quote**:\n"
        "   - Use speech bubbles to understand the narrative’s context but avoid quoting or referencing them directly. Focus on summarizing the character actions, emotions, and plot developments instead.\n"
        "   - **Example**: If a character expresses fear in a dialogue, describe their body language and facial expressions, and how these emotions impact the story.\n\n"

        "4. **Seamless Transitions Between Arcs**:\n"
        "   - Transitions between arcs should be smooth and logical. If a panel contributes to an ongoing arc, integrate it into the existing arc. If the story shifts (e.g., a new conflict or setting), start a new arc, ensuring it picks up logically from where the previous one ended.\n"
        "   - **Example**: After a tense conversation, if a character leaves to confront an enemy, the arc should transition naturally from the conversation to the confrontation.\n\n"

        "5. **Refine and Update the Summary**:\n"
        "   - With each new page or panel, incrementally improve the narrative summary. You may need to expand, merge, or adjust arcs based on new information or clarify details from previous panels.\n"
        "   - Align arcs to build a single, cohesive timeline, ensuring that no unrelated or disjointed content interrupts the flow.\n\n"

        "6. **Keep Descriptions Concise and Focused**:\n"
        "   - Focus on describing key plot points, character actions, and emotional shifts. Avoid excessive visual details unless they are crucial to the plot or character development.\n"
        "   - **Example**: If a character retrieves an object of significance, describe the action and its importance, rather than delving into unnecessary background details.\n\n"

        "Additional Guidelines:\n"
        "- **No Gaps in Arcs**:\n"
        "  - Ensure that each narrative arc runs continuously, with no holes or skipped panels. If a panel belongs to a given arc, ensure it fits within the timeline and directly connects to the previous and subsequent panels in that arc.\n\n"

        "- **Chronological Alignment**:\n"
        "  - All arcs must fit into a single, linear timeline, progressing logically from one event to the next. If a new arc starts, it must pick up directly after the last one ended, without revisiting or overlapping previous content.\n\n"

        "- **No Overlapping Arcs**:\n"
        "  - Each arc should be distinct from others. Do not allow two arcs to describe different parts of the same scene or moment. Once an arc ends, it cannot reappear in another form.\n\n"

        "Tone and Style:\n"
        "- **Novel-like Descriptions**:\n"
        "  - Treat the task as if you are writing scenes from a novel, translating visual comic events into a cohesive, flowing narrative.\n\n"

        "- **Active Voice**:\n"
        "  - Use active voice to keep the narrative lively, focusing on actions and emotional intensity.\n\n"

        "- **Stick to Clear Depictions**:\n"
        "  - Do not speculate or add interpretations unless they are clearly implied by the comic panels. Stick to describing what is explicitly depicted in the story.\n\n"

        "Task:\n"
        "Your job is to update the summary and chapter list based on new panels or pages. Focus on expanding and refining existing narrative arcs based on new developments, or create a new arc if the story shifts significantly. On average there should be no Scene that is longer than 2 pages. Also add a list of chracters that appear in this scene. If there is no clue about the direct name of one chracter, try to describe hime like Officer or James Mother.\n\n"

        "- Each arc should form part of a continuous timeline, with no gaps or skipped sections.\n"
        "- Ensure that arcs are aligned and do not overlap. Once an arc concludes, the next one should follow it directly.\n"
        "- When new panels clarify or enhance earlier parts of the story, update the summary to reflect this, ensuring a smooth, chronological progression.\n\n"

        "Bounding Boxes for Panel Coordinates:\n"
        "For each new narrative arc, output the coordinates (bounding boxes) of the panel that marks the start of the arc. The coordinates should be in the format: (x1, y1, x2, y2) where (x1, y1) is the top-left corner of the panel, and (x2, y2) is the bottom-right corner. You get all panel data as an Input and try to find the most suitable panel bounding box.\n\n"

        "Output Format:\n"
        "{{\n"
        "  \"Narrative_Arcs\": [\n"
        "    {{\n"
        "      \"arc_id\": \"1\",\n"
        "      \"title\": \"Title of Arc\",\n"
        "      \"starting_page\": \"1\",\n"
        "      \"description\": \"Description of Arc\",\n"
        "      \"occurring_characters\": \"[\"(James, Police Officer)\"]\",\n"
        "      \"coordinates_of_starting_panel\": \"(x1, y1, x2, y2)\",\n"
        "      \"panel_index_of_startpanel\": \"1\"\n"
        "      \"reasoning_for_new_arc\": \"explanation text\"\n"
        "      \"confidence\": \" ...\"\n"
        "    }}\n"
        "    ...\n"
        "  ]\n"
        "}}\n\n"

        "Each iteration should update and expand upon the previous narrative arcs. If an arc spans multiple panels, expand on it. If new panels provide clarity or enhance earlier parts of the narrative, update the entire summary to reflect this.\n"
        "Also, try to summarize smaller arcs into larger ones for better coherence."
        "Keep in mind that a new arc can only start at the end of the previous arc. Provide the page index in which the arc starts. The page index is always given as an extra input."
        "Also add a confidence which tells how sure you are if an arc is selfstanding."
    )

    def __init__(self, name: str, volume: int, main_series: Series, secondary_series: Optional[List[Series]],
                 rgb_arrays):
        page_pairs = self.convert_array_to_page_pairs(rgb_arrays)
        self.current_comic = self.convert_images_to_comic(name, volume, main_series, secondary_series, page_pairs)
        self.model = AutoModel.from_pretrained("ragavsachdeva/magiv2", trust_remote_code=True).eval()
        self.tagger = wdv3_timm.Tagger(model_name='vit', gen_threshold=0.35, char_threshold=0.75)
        model_name: str = "gemini-1.5-flash-002"
        self.model_gemini = genai.GenerativeModel(model_name=model_name)
        genai.configure(api_key="AIzaSyCoUfTjZU-zNZ2lNKY_BnDuyNTu8lHQ9EM")
        self.detect_panels()
        self.describe_narrative()
        self.match_entities()
        self.create_tags()
        #self.find_clusters()

    @staticmethod
    def convert_images_to_comic(name: str, volume: int, main_series: Series,
                                secondary_series: Optional[List[Series]], page_pairs):
        return Comic(name, volume, main_series, secondary_series, page_pairs)

    def guess_speakers(self):
        for page_pair in self.current_comic.page_pairs:
            for page in page_pair:
                if not page:
                    continue
                for panel in page.panels:
                    t = 2

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
            'content': self.comic_summarization_prompt,
            'images': [image_bytes]
        }

        try:
            res = ollama.chat(
                model='llava',
                messages=[message]
            )
        except Exception as e:
            print(e)

        description = res['message']['content']
        print(description)
        return description

    def describe_narrative(self):
        """
        Summarizes the comic pages and integrates the previous narrative context.
        :param pages: List of comic page objects that contain images.
        :return: A cumulative summary of the entire comic.
        """
        overall_summaries = []
        current_summary = ""
        page_counter = 1
        pages_list = []
        for pages in self.current_comic.page_pairs:
            for page in pages:
                if page is None:
                    continue

                print('Processing new page...')

                pages_list.append(page)

                image_rgb = cv2.cvtColor(page.page_image, cv2.COLOR_BGR2RGB)

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                    temp_filename = tmp_file.name
                    cv2.imwrite(temp_filename, image_rgb)

                sample_file = genai.upload_file(path=temp_filename,
                                                display_name="Comic Page")

                page_information = 'Current Page Data: \n'
                page_information = page_information + f'Page Number: + {page.page_index} \n Panel Data: \n'
                panels = page.panels

                for i, panel in enumerate(panels):
                    page_information = page_information + f'Panel Index: + {i + 1}  Boundingbox = {panel.bounding_box} \n'

                previous_summary_text = f"\n\nPrevious Narrative Summary:\n{current_summary}" if current_summary else ""
                full_prompt = self.comic_summarization_prompt + '\n' + previous_summary_text + '\n' + page_information

                try:
                    response = self.model_gemini.generate_content([sample_file, full_prompt])
                except Exception as e:
                    print(e)

                description = response.text

                current_summary += f'Page: {page_counter} \n {description} \n'

                overall_summaries.append(description)

                print(f"Page {pages.index(page) + 1} Summary:\n{description}\n")

                page_counter += 1

        cleaned_json_string = response.text.strip('```json')
        cleaned_json_string = cleaned_json_string.strip('```')
        cleaned_json_string = cleaned_json_string.replace('{{', '{')
        cleaned_json_string = cleaned_json_string.replace('}}', '}')
        cleaned_json_string = cleaned_json_string.strip('```')
        try:
            json_object = json.loads(cleaned_json_string)
        except Exception as e:
            print(e)
            full_prompt = f"Transform this JSON in valid json that can be used in python: {cleaned_json_string}. Only return the JSON object. this was the error message {e}"
            response = self.model_gemini.generate_content(full_prompt)
            cleaned_json_string = response.text.strip('```json')
            cleaned_json_string = cleaned_json_string.strip('```')
            print('new generated json: ' + cleaned_json_string)
            try:
                json_object = json.loads(cleaned_json_string)
            except Exception as e:
                print(e)
                print("Error in the scene detection. Please start again or manually select the scenes.")
                return

        for arc in json_object['Narrative_Arcs']:
            page = pages_list[int(arc['starting_page']) - 1]
            bounding_box = arc['coordinates_of_starting_panel']

            bounding_box = list(map(float, bounding_box.strip("()").split(", ")))

            self.find_nearest_bounding_box(bounding_box,page.panels)
        self.current_comic.update_scenes()

        for scene in self.current_comic.scenes:
            for panel in scene:
                print(panel.scene_id)

        self.current_comic.scene_data = cleaned_json_string

    def find_nearest_bounding_box(self,bounding_box, panels):
        target_x, target_y, target_width, target_height = bounding_box
        target_center = (target_x + target_width / 2, target_y + target_height / 2)

        min_distance = float('inf')
        nearest_panel_index = None

        for i,panel in enumerate(panels):
            panel_box = panel.bounding_box
            panel_center = (panel_box['x'] + panel_box['width'] / 2, panel_box['y'] + panel_box['height'] / 2)

            distance = ((target_center[0] - panel_center[0]) ** 2 + (target_center[1] - panel_center[1]) ** 2) ** 0.5

            if distance < min_distance:
                min_distance = distance
                nearest_panel_index = i

        panels[nearest_panel_index].starting_tag = True

    def find_names(self):
        for pages in self.current_comic.page_pairs:
            for page in pages:
                if page is None:
                    continue

                notated_img = page.annotated_image(True, False, True)

    def match_entities(self):
        self.current_comic.reset_entities()
        scene_images = self.current_comic.get_scene_images()
        for scene_image, used_pages in scene_images:
            data = self.detect_objects(scene_image)
            panel_list = data[0]['panels']
            speechbubble_list = data[0]['texts']
            entity_list = data[0]['characters']
            tails = data[0]['tails']
            text_character_associations = data[0]['text_character_associations']
            text_tail_associations = data[0]['text_tail_associations']
            character_cluster_labels = data[0]['character_cluster_labels']
            is_essential_text = data[0]['is_essential_text']
            character_names = data[0]['character_names']

            entities_per_page = self.adjust_bounding_boxes(entity_list, used_pages, character_cluster_labels)

            for i,entities in enumerate(entities_per_page):
                for bbox,cluster_id in entities:
                    page = used_pages[i]
                    entity = Entity(bbox)
                    entity.named_entity_id = cluster_id
                    entity.image = iu.image_from_bbox(page.page_image, bbox)

                    for panel in page.panels:
                        best_panel = None
                        highest_iou = 0

                        for panel in page.panels:
                            iou_value = iu.calculate_iou(panel.bounding_box, entity.bounding_box)

                            if iou_value > highest_iou:
                                highest_iou = iou_value
                                best_panel = panel

                        if best_panel is not None and highest_iou > 0:
                            if not any(iu.calculate_iou(existing_entity.bounding_box, entity.bounding_box) > 0.7
                                       for existing_entity in best_panel.entities):
                                best_panel.entities.append(entity)

                        for panel in page.panels:
                            entities = panel.entities
                            panel.entities = sorted(entities,
                                                    key=lambda p: ((p.bounding_box['y'] - p.bounding_box['height']),
                                                                   (p.bounding_box['x']) - p.bounding_box['width']))

    def adjust_bounding_boxes(self, bounding_boxes, scene_pages, character_cluster_labels):
        page_offsets = []
        x_offset = 0

        for page in scene_pages:
            page_offsets.append(x_offset)
            x_offset += page.page_image.shape[1]

        adjusted_boxes_by_page = [[] for _ in scene_pages]

        for cluster,box in enumerate(bounding_boxes):
            bbox = self.x1y1x2y2_to_xywh(box)
            x, y, w, h = bbox

            page_index = None
            for i, offset in enumerate(page_offsets):
                if bbox['x'] >= offset and bbox['x'] < offset + scene_pages[i].page_image.shape[1]:
                    page_index = i
                    break

            if page_index is None:
                print("Bounding box does not fit within any page range.")
                continue

            bbox['x']= bbox['x'] - page_offsets[page_index]
            adjusted_boxes_by_page[page_index].append((bbox, character_cluster_labels[cluster]))

        return adjusted_boxes_by_page

    #TODO: delete, but keep for example of cluster size estimation for the thesis
    def find_clusters(self):
        entity_tag_str = "List of Entities: \n"
        entity_counter = 1
        panel_counter = 1
        for scene in self.current_comic.scenes:
            for panel in scene:
                entity_tag_str += f"Panel {panel_counter}: \n"
                for entity in panel.entities:
                    entity_tag_str += f"Entity {entity_counter} Tags: \n"
                    for tag, confidence in entity.tags:
                        entity_tag_str += f"Tag: {tag}, Confidence: {confidence} \n"
                    entity_counter += 1
                panel_counter += 1
            prompt = (
                "Given a list of Entities that occur in a Comic and their Tags with confidence, try to find the ammount of characters that are in there. This means you should return a cluster size. Keep in mind that in each panel the same entity can only apear once. \n"
                "Output format: Clustersize: 1... \n"
                "Calulate a clustersize estimate for the given list. \n")

            entity_tag_str += prompt
            #prompt += entity_tag_str

            print(entity_tag_str)

            response = self.model_gemini.generate_content([entity_tag_str])

            print(response.text)

            entity_tag_str = "List of Entities: \n"

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

    #Use local model
    def handle_create_tags(self, page):

        for panel in page.panels:
            for entity in panel.entities:
                pil_image = Image.fromarray(entity.image)

                pil_image.save('tmp.jpg')
                image_input = handle_file('tmp.jpg')

                result = self.tagger.process_image('tmp.jpg')
                confidences = result
                tag_confidence_tuples = [(label, confidence) for label, confidence in confidences.items()]

                entity.tags = tag_confidence_tuples

                # TODO: add to .env as name
                os.remove('tmp.jpg')
                if not self.is_character(entity.tags):
                    panel.entities.remove(entity)

    def is_character(self, tag_confidence_tuples, threshold=0.7) -> bool:
        # Filter tags based on the threshold
        valid_tags = {tag: score for tag, score in tag_confidence_tuples if score >= threshold}

        # Check if the tag 'multiple_boys' exists in the valid tags and adjust logic based on the threshold
        if "multiple_boys" in valid_tags and valid_tags["multiple_boys"] > 0.7:
            return False  # This entity should not be considered a character (based on the 'multiple_boys' tag)

        # Check if valid tags contain any of the non-character tags
        #if len(valid_tags) > 0 and all(
        #tag in ["no_humans", "english_text", "monochrome", "greyscale"] for tag in valid_tags.keys()):
        #tag in ["no_humans"] for tag in valid_tags.keys()):
        #return False  # The entity is not a character if all valid tags are non-character tags

        # If we have valid tags, and none of the conditions return False, we return True (it's a character)
        return len(valid_tags) > 0

    #TODO: Refactor with extract_speech_bubbles
    def detect_panels(self):

        for i, page_pair in enumerate(self.current_comic.page_pairs):
            for page in page_pair:
                if not page:
                    continue

                logging.debug(f'Starting Panel Extraction thread for page {page.page_index}')
                self.handle_detect_objects(page)

                logging.debug(f'Threads for Panel Extraction of page_pair {i} finished')
            logging.debug('\n')

    def x1y1x2y2_to_xywh(self, bbox):
        x1, y1, x2, y2 = bbox
        return {
            "x": max(x1, 0),
            "y": max(y1, 0),
            "width": max(x2 - x1, 0),
            "height": max(y2 - y1, 0)
        }

    def handle_detect_objects(self, page: Page):
        data = self.detect_objects(page.page_image)
        panel_list = data[0]['panels']
        speechbubble_list = data[0]['texts']
        entity_list = data[0]['characters']
        tails = data[0]['tails']
        text_character_associations = data[0]['text_character_associations']
        text_tail_associations = data[0]['text_tail_associations']
        character_cluster_labels = data[0]['character_cluster_labels']
        is_essential_text = data[0]['is_essential_text']
        character_names = data[0]['character_names']

        panels = []

        for panel in panel_list:
            bbox = self.x1y1x2y2_to_xywh(panel)
            print(bbox)
            panel_image = iu.image_from_bbox(page.page_image, bbox)

            #description = self.describe_image(panel_image)
            description = ""
            panel = Panel(description, bbox, panel_image)
            panel.page_id = page.page_index
            panel.descriptions.append(panel.description)
            panels.append(panel)

        panels = sorted(panels, key=lambda p: ((p.bounding_box['y'] - p.bounding_box['height']),
                                               (p.bounding_box['x']) - p.bounding_box['width']))
        page.panels = panels

        for i, entity_ in enumerate(entity_list):
            bbox = self.x1y1x2y2_to_xywh(entity_)
            entity = Entity(bbox)
            entity.named_entity_id = character_cluster_labels[i]
            entity.image = iu.image_from_bbox(page.page_image, bbox)

            for panel in page.panels:
                best_panel = None
                highest_iou = 0

                for panel in page.panels:
                    iou_value = iu.calculate_iou(panel.bounding_box, entity.bounding_box)

                    if iou_value > highest_iou:
                        highest_iou = iou_value
                        best_panel = panel

                if best_panel is not None and highest_iou > 0:
                    if not any(iu.calculate_iou(existing_entity.bounding_box, entity.bounding_box) > 0.7
                               for existing_entity in best_panel.entities):
                        best_panel.entities.append(entity)

                for panel in page.panels:
                    entities = panel.entities
                    panel.entities = sorted(entities,
                                            key=lambda p: ((p.bounding_box['y'] - p.bounding_box['height']),
                                                           (p.bounding_box['x']) - p.bounding_box['width']))

        essential_counter = 0
        for i, speechbubble in enumerate(speechbubble_list):
            bbox = self.x1y1x2y2_to_xywh(speechbubble)
            speech_bubble_image = iu.image_from_bbox(page.page_image, bbox)
            description = ""
            speech_bubble = SpeechBubble(SpeechBubbleType.SPEECH, description, bbox, speech_bubble_image)
            speech_bubble.type = 'dialogue'
            speech_bubble.person_list = []
            #TODO: Use classification of speech bubble for this
            if is_essential_text[i]:
                #speech_bubble.speaker_id = character_cluster_labels[text_character_associations[essential_counter][1]]
                essential_counter += 1
                speech_bubble.speaker_id = 1
                #else:
                speech_bubble.speaker_id = 0

            for panel in page.panels:
                if iu.is_bbox_overlapping(panel.bounding_box, speech_bubble.bounding_box):
                    panel.speech_bubbles.append(speech_bubble)

                for panel in page.panels:
                    speech_bubbles = panel.speech_bubbles
                    panel.speech_bubbles = sorted(speech_bubbles,
                                                  key=lambda p: ((p.bounding_box['y'] - p.bounding_box['height']),
                                                                 (p.bounding_box['x']) - p.bounding_box['width']))

    def detect_objects(self, image):
        images = []
        images.append(image)
        character_bank = {
            "images": [],
            "names": []
        }
        with torch.no_grad():
            page_results = self.model.do_chapter_wide_prediction(images, character_bank, use_tqdm=True,
                                                                 do_ocr=False)
            print(page_results)
            #for i, (image, page_result) in enumerate(zip(image, page_results)):
            #self.model.visualise_single_image_prediction(image, page_result, f"page_{random.randint(0, 10000)}.png")

        return page_results

    def canny_panels(self, image):
        grayscale = rgb2gray(image)
        edges = canny(grayscale)

        thick_edges = dilation(dilation(edges))
        segmentation = ndi.binary_fill_holes(thick_edges)

        labels = label(segmentation)

        def do_bboxes_overlap(a, b):
            return (
                    a[0] < b[2] and
                    a[2] > b[0] and
                    a[1] < b[3] and
                    a[3] > b[1]
            )

        def merge_bboxes(a, b):
            return (
                min(a[0], b[0]),
                min(a[1], b[1]),
                max(a[2], b[2]),
                max(a[3], b[3])
            )

        regions = regionprops(labels)
        panels = []

        for region in regions:
            for i, panel in enumerate(panels):
                if do_bboxes_overlap(region.bbox, panel):
                    #panels[i] = merge_bboxes(panel, region.bbox)
                    panels.append(region.bbox)
                    break
            else:
                panels.append(region.bbox)

        im = image
        for i, bbox in reversed(list(enumerate(panels))):
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if area < 0.01 * im.shape[0] * im.shape[1]:
                del panels[i]

        formatted_panels = []
        for bbox in panels:
            x = bbox[1]
            y = bbox[0]
            width = bbox[3] - bbox[1]
            height = bbox[2] - bbox[0]
            formatted_panels.append({
                "x": x,
                "y": y,
                "width": width,
                "height": height
            })

        return formatted_panels
