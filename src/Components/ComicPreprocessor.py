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
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.morphology import dilation
from scipy import ndimage as ndi
from skimage.measure import label
from skimage.color import label2rgb
from skimage.measure import regionprops
import imageio
from PIL import Image,ImageDraw,ImageFont
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
        "Your job is to update the summary and chapter list based on new panels or pages. Focus on expanding and refining existing narrative arcs based on new developments, or create a new arc if the story shifts significantly.\n\n"

        "- Each arc should form part of a continuous timeline, with no gaps or skipped sections.\n"
        "- Ensure that arcs are aligned and do not overlap. Once an arc concludes, the next one should follow it directly.\n"
        "- When new panels clarify or enhance earlier parts of the story, update the summary to reflect this, ensuring a smooth, chronological progression.\n\n"

        "Bounding Boxes for Panel Coordinates:\n"
        "For each new narrative arc, output the coordinates (bounding boxes) of the panel that marks the start of the arc. The coordinates should be in the format: (x1, y1, x2, y2) where (x1, y1) is the top-left corner of the panel, and (x2, y2) is the bottom-right corner.\n\n"

        "Output Format:\n"
        "{{\n"
        "  \"Narrative_Arcs\": [\n"
        "    {{\n"
        "      \"arc_id\": \"1\",\n"
        "      \"title\": \"Title of Arc\",\n"
        "      \"pages\": [1, 2],\n"
        "      \"description\": \"Description of Arc\",\n"
        "      \"coordinates\": \"(x1, y1, x2, y2)\"\n"
        "    }}\n"
        "    ...\n"
        "  ]\n"
        "}}\n\n"

        "Each iteration should update and expand upon the previous narrative arcs. If an arc spans multiple panels, expand on it. If new panels provide clarity or enhance earlier parts of the narrative, update the entire summary to reflect this.\n"
        "Also, try to summarize smaller arcs into larger ones for better coherence."
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
        #self.create_tags()
        print(self.describe_narrative())

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

        res = ollama.chat(
            model='llava',
            messages=[message]
        )

        description = res['message']['content']
        print(description)
        return description

    def describe_narrative(self):
        """
        Summarizes the comic pages and integrates the previous narrative context.
        :param pages: List of comic page objects that contain images.
        :return: A cumulative summary of the entire comic.
        """
        overall_summaries = []  # To hold all scene summaries
        current_summary = ""  # Holds the summary for the current iteration
        page_counter = 1
        for pages in self.current_comic.page_pairs:  # Assuming page_pairs is part of the comic object
            for page in pages:
                if page is None:
                    continue

                print('Processing new page...')

                # Convert the page image to RGB
                image_rgb = cv2.cvtColor(page.page_image, cv2.COLOR_BGR2RGB)

                # Save image as temporary file instead of converting to byte array
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                    temp_filename = tmp_file.name
                    cv2.imwrite(temp_filename, image_rgb)  # Save image to temporary file

                sample_file = genai.upload_file(path=temp_filename,
                                                display_name="Comic Page")

                # Combine the current summary into the prompt
                previous_summary_text = f"\n\nPrevious Narrative Summary:\n{current_summary}" if current_summary else ""
                full_prompt = self.comic_summarization_prompt + previous_summary_text

                # Prepare the image and prompt for the model
                response = self.model_gemini.generate_content([sample_file, full_prompt])

                # Get the description from the model's response
                description = response.text

                # Update the current summary with the latest description
                current_summary += f'Page: {page_counter} \n {description} \n'

                # Append the new description to the overall summaries for continuity
                overall_summaries.append(description)

                # Output the current page's summary for review
                print(f"Page {pages.index(page) + 1} Summary:\n{description}\n")

                page_counter += 1

        # After processing all pages, return the cumulative summary
        return response.text  # Returns the final cumulative summary at the end

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

        print(text_character_associations)
        print(is_essential_text)
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
                speech_bubble.speaker_id= 1
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
