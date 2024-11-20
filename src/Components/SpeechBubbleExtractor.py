import json
import os
import tempfile
from collections import Counter

import cv2
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
import google.generativeai as genai
from openai import OpenAI

from src.Components import ComicPreprocessor

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class SpeechBubbleExtractor:
    nvidia_key = os.getenv('APIKEY_NVIDIA')
    comic_panel_prompt = """
    Given a list of comic panels, including descriptions, previous context (with assumed speakers, their dialogue, and character profiles), and text extracted from speech bubbles, summarize the events happening in each panel. For each panel, identify the most likely speaker for each speech bubble by considering existing character profiles or inferring from context. Establish a connection between each panel and the previous panels to create a coherent narrative. Finally, create or update character profiles based on their actions and dialogue across the panels.
    
    Additional Informations:
    -If a name appears in a text, its very unlikely that its the name of the speaker, but more likely that its the name of a conversation partner.
    -Try to find aliases, If for example Peter, Peter Parker and Pete exists, they are most likely the same person.
    -You are allowed to change previous informations if you gained new insights about speaker prediction.
    
    Provide the output in the following JSON format:

    {
      "panels": [
        {
          "panel_number": 1,
          "summary": "A brief description of the panel's events.",
          "dialogue": [
            {
              "speechbubble_id": "Index of the speechbubble",
              "text": "The exact speech bubble text.",
              "type": "Type of text. For example: Speech, Inner Monologue."
              "predicted_speaker": "The predicted speaker based on context."
            }
          ],
          "connection_to_previous_panel": "Explanation of how this panel connects to the prior panels."
        },
        ...
      ],
      "character_profiles": {
        "Character Name": {
          "description": "A description of the character based on their role and actions.",
          "notable_traits": "Key traits, motivations, or patterns observed in their behavior.",
          "relationships": "Details on how this character relates to others in the story."
        },
        ...
      }
    }

    ### Example Input:
    #### Previous Context:
  {
      "panels": [
        {
          "panel_number": 1,
          "summary": "A superhero flies over a city at night with a glowing object as civilians look on in awe. A villain observes from a rooftop, smirking confidently. The superhero is clearly pursuing the villain after a theft.",
          "dialogue": [
            {
              "speechbubble_id": "1",
              "text": "You'll never get away with this!",
              "type": "Speech",
              "predicted_speaker": "Superhero"
            },
            {
              "speechbubble_id": "2",
              "text": "Oh, but I already have!",
              "type": "Speech",
              "predicted_speaker": "Villain"
            }
          ],
          "connection_to_previous_panel": "This panel establishes the ongoing chase between the superhero and the villain."
        }
      ],
      "character_profiles": {
        "Superhero": {
          "description": "A determined and morally upright individual dedicated to stopping the villain.",
          "notable_traits": "Courageous, relentless in pursuit, motivated by justice.",
          "relationships": "The primary adversary of the Villain."
        },
        "Villain": {
          "description": "A cunning and confident thief who wields a stolen artifact of great power.",
          "notable_traits": "Clever, arrogant, resourceful.",
          "relationships": "The main opponent of the Superhero, taunts them frequently."
        }
      }
    }

    #### Panel Description:
    The superhero lands on the rooftop to confront the villain. The villain draws a weapon, glowing with the same energy as the stolen artifact.

    #### Speech Bubble Texts:
    "Panel Number: 2"
    "Speechbubble Data:"
    "Speechbubble Index: 1 Text: This ends now!"
    "Speechbubble Index: 2 Text: Not if I end you first!"
    
    ### Example Output:
    {
      "panels": [
        {
          "panel_number": 1,
          "summary": "A superhero flies over a city at night with a glowing object as civilians look on in awe. A villain observes from a rooftop, smirking confidently. The superhero is clearly pursuing the villain after a theft.",
          "dialogue": [
            {
              "speechbubble_id": "1",
              "text": "You'll never get away with this!",
              "type": "Speech",
              "predicted_speaker": "Superhero"
            },
            {
              "speechbubble_id": "2",
              "text": "Oh, but I already have!",
              "type": "Speech",
              "predicted_speaker": "Villain"
            }
          ],
          "connection_to_previous_panel": "This panel establishes the ongoing chase between the superhero and the villain."
        },
        {
          "panel_number": 2,
          "summary": "The superhero confronts the villain on the rooftop, ready for a fight. The villain reveals a weapon powered by the stolen artifact.",
          "dialogue": [
            {
              "speechbubble_id": "1",
              "text": "This ends now!",
              "type": "Speech",
              "predicted_speaker": "Superhero"
            },
            {
              "speechbubble_id": "2",
              "text": "Not if I end you first!",
              "type": "Speech",
              "predicted_speaker": "Villain"
            }
          ],
          "connection_to_previous_panel": "The confrontation follows the chase established in the first panel. The villain shows they are armed with the stolen artifact's power."
        }
      ],
      "character_profiles": {
        "Superhero": {
          "description": "A determined and morally upright individual dedicated to stopping the villain.",
          "notable_traits": "Courageous, relentless in pursuit, motivated by justice.",
          "relationships": "The primary adversary of the Villain."
        },
        "Villain": {
          "description": "A cunning and confident thief who wields a stolen artifact of great power.",
          "notable_traits": "Clever, arrogant, resourceful.",
          "relationships": "The main opponent of the Superhero, taunts them frequently."
        }
      }
    }

    Now, process the following input:
    """
    novelization_prompt = """
    Instructions: You are a skilled writer tasked with transforming the descriptions of comic panels and speech bubbles into a cohesive, engaging narrative. Your goal is to preserve the tone, pacing, and emotions of the scenes while adapting them into prose that reads like a novel. The resulting story should be immersive and maintain the integrity of the original comic. Use descriptive language to paint vivid images, capture character emotions and dynamics, and integrate dialogue naturally.

    Constraints:
    1. You are not allowed to rename characters from the input. Names should be preserved unless there is an obvious error or misnaming in the input data that needs correction.
    2. Multiple scenes should be thoughtfully combined into larger, interconnected chapters to create a seamless narrative flow. Each scene is not necessarily equal to a chapter; instead, combine related scenes to form cohesive chapters.
    3. The story should remain faithful to the input descriptions, capturing all provided details, but it should also feel natural and engaging in its transitions and pacing.
    4. Use descriptive language to evoke vivid imagery, emotions, and tension without adding unnecessary details that detract from the original tone and intent. But do not add too much descriptions.
    5. Do not change the speaker of speech bubbles.
    
    Example rename of characters or speech bubbles:
        Example 1:
            Original input: 
                Max: Hey Max, how are you doing?
                Karl: Oh hello Karl, im fine and you?
            Correction:
                Karl: Hey Max, how are you doing?
                Max: Oh hello Karl, im fine and you?
        Example 2:
            Original input:
                James: I will defeat you ...
                Jackson: ... but not today.
            Correction:
                James: I will defeat you ...
                James: ... but not today.

    Example Input:
    Scene 1  
    Panel 1: A panoramic view of a sprawling city at night, its lights twinkling like a vast constellation. The perspective is from a high vantage point, overlooking a snow-covered mountainside which takes up the lower half of the panel.  
    Narrator: PROLOGUE  

    Scene 2  
    Panel 1: A car drives along a winding mountain road at night. The road is carved into the side of a cliff, with a cityscape visible in the distance.  
    Driver: We're supposed to pick up Navin at eight o'clock. We're late.  

    Panel 2: The interior of the car is shown, displaying a digital clock that reads 7:45. Someone in the car reassures the driver that they have time.  
    Passenger: We have plenty of time. At least half an hour.  
    Clock: 7:45  

    Example Output:
    Prologue  
    The city stretched beneath the velvet expanse of night like a scatter of stars, each light a testament to lives intertwined and dreams in motion. From the mountainside, its snow-laden slopes a stark contrast to the luminous sprawl, the world below seemed distant—untouchable. It was the perfect place to begin a story that would unravel the fragile line between safety and peril.  

    Chapter 1: The Road to Navin  
    The car hugged the curves of the mountain road, its headlights slicing through the dark. Far below, the city sparkled with indifferent brilliance, its towers mere pinpricks of light from this height.  

    "We're supposed to pick up Navin at eight o'clock," the driver said, his voice tinged with exasperation. He tightened his grip on the wheel. "We're late."  

    Beside him, his passenger—a child with an unconcerned air—glanced at the dashboard. The digital clock glowed an unrelenting 7:45.  
    "We have plenty of time," they replied breezily. "At least half an hour."  

    The driver shot them a skeptical look. "Fifteen minutes," he corrected with a sigh, "is not half an hour."  

    The child smirked, leaning back in their seat as though settling into a debate they’d already won. "I think Dad just lives in an alternate universe," they mused, their voice carrying a note of dry amusement.  

    Scene Formatting for Novelization:  
    1. Panel Descriptions → Detailed narrative prose (focus on setting, action, and emotion).  
    2. Dialogue → Integrated into prose as natural exchanges between characters.  
    3. Sound Effects or Actions → Incorporated into the narrative to maintain pacing and drama.  
    4. Emotion and Subtext → Expanded through internal thoughts or descriptive language.  
    5. Scene Connections → Create fluid transitions between scenes and sentences to ensure a seamless, cohesive narrative flow.  

    Your Turn:  
    Using the example and formatting provided, novelize the following scene descriptions into a cohesive collection of chapters. Be sure to capture the suspense, relationships, and mood as faithfully as possible while adhering to the constraints outlined.
    """

    def __init__(self, current_comic: Comic, comic_preprocessor: ComicPreprocessor):
        self.current_comic = current_comic
        self.comic_preprocessor = comic_preprocessor
        self.extract_speech_bubbles()
        self.speaker_association()
        self.story_creation()

    #TODO: Refactor with detect_panel
    def extract_speech_bubbles(self):
        comiq.set_api_key(self.comic_preprocessor.apikey)
        for i, page_pair in enumerate(self.current_comic.page_pairs):
            for page in page_pair:
                if not page:
                    continue
                print(f'Starting Speechbubble Extraction thread for page {page.page_index}')
                self.handle_extract_speech_bubbles(page)


    def story_creation(self):
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=self.nvidia_key
        )
        previous_summarization = ""
        for i,scene in enumerate(self.current_comic.scenes):
            completion = client.chat.completions.create(
                    model="nvidia/llama-3.1-nemotron-70b-instruct",
                    messages=[{"role": "user", "content": self.novelization_prompt + self.current_comic.to_narrative() + f"\n Only process Scene number {i+1}. Output only the novelazation of Scene {i+1} but keep the whole story in mind and this summary of previous scenes:" + previous_summarization}],
                    temperature=0.5,
                    top_p=1,
                    max_tokens=1024,
                    stream=True
                )


            for chunk in completion:
                if chunk.choices[0].delta.content is not None:

                    print(chunk.choices[0].delta.content, end="")
                    previous_summarization += chunk.choices[0].delta.content
            previous_summarization += "\n"
        with open("generated_story.txt", "w", encoding="utf-8") as file:
            file.write( previous_summarization)

    def speaker_association(self):
        for scene in self.current_comic.scenes:
            current_summary = ""
            for i,panel in enumerate(scene):
                image_rgb = cv2.cvtColor(panel.image, cv2.COLOR_BGR2RGB)

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                    temp_filename = tmp_file.name
                    cv2.imwrite(temp_filename, image_rgb)

                sample_file = genai.upload_file(path=temp_filename,
                                                display_name="Comic Panel")

                panel_information = 'Current Panel Data: \n'
                panel_information = panel_information + f'Panel Number: + {i+1} \n Speechbubble Data: \n'
                speechbubbles = panel.speech_bubbles

                for i, speechbubble in enumerate(speechbubbles):
                    panel_information = panel_information + f'Speechbubble Index: + {i + 1}  Text: {speechbubble.text} \n'

                previous_summary_text = f"\n\nPrevious Narrative Summary:\n{current_summary}" if current_summary else ""
                full_prompt = self.comic_panel_prompt + '\n' + previous_summary_text + '\n' + panel_information

                try:
                    response = self.comic_preprocessor.model_gemini.generate_content([sample_file, full_prompt])
                except Exception as e:
                    print(e)

                description = response.text

                current_summary +=  description + "\n"

                print(f"Panel {i+1} Speaker:\n{description}\n")

                cleaned_json_string = response.text.replace('```json', '').replace('```', '').strip()

                try:
                    json_object = json.loads(cleaned_json_string)
                except Exception as e:
                    print(f"Initial JSON parsing error: {e}")
                    full_prompt = (
                        f"Transform this JSON into valid JSON that can be used in Python: {cleaned_json_string}. "
                        f"Only return the JSON object. This was the error message: {e}"
                    )
                    response = self.comic_preprocessor.model_gemini.generate_content(full_prompt)
                    try:
                        cleaned_json_string = response.text.replace('```json', '').replace('```', '').strip()
                        print(f"Corrected JSON: {cleaned_json_string}")
                        json_object = json.loads(cleaned_json_string)
                    except Exception as e:
                        print(f"Error parsing corrected JSON: {e}")
                        print("Error in processing JSON. Please try again.")
                        return

                try:
                    for panel_data in json_object['panels']:
                        panel_index = int(panel_data['panel_number']) - 1
                        if panel_index < 0 or panel_index >= len(scene):
                            print(f"Invalid panel number: {panel_data['panel_number']}")
                            continue

                        panel = scene[panel_index]
                        panel.description = panel_data['summary']

                        for bubble_data in panel_data['dialogue']:
                            bubble_id = int(bubble_data['speechbubble_id'])-1
                            if bubble_id < len(panel.speech_bubbles):
                                bubble = panel.speech_bubbles[bubble_id]
                                bubble.text = bubble_data['text']
                                bubble.type = bubble_data['type']
                                bubble.speaker_id = bubble_data['predicted_speaker']
                                print(bubble.get_string() + "\n")
                            else:
                                print(f"Invalid bubble ID: {bubble_id}")
                except Exception as e:
                    print(f"Error processing panels: {e}")

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
            if not description:
                continue
            person_list = self.apply_named_entitiy_recognition(description, nlp)

            speech_bubble = SpeechBubble(speech_bubble_type, description, bounding_box, speech_bubble_image)

            # Restore old metadata if possible
            for i, panel in enumerate(page.panels):
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

            # Determine the best panel exclusively
            best_panel = None
            highest_overlap = 0

            for panel in page.panels:
                overlap = iu.calculate_overlap_percentage(speech_bubble.bounding_box, panel.bounding_box)
                if overlap > highest_overlap:
                    highest_overlap = overlap
                    best_panel = panel

            # Assign speech bubble to the best panel only
            if best_panel and highest_overlap > 0.1:  # Use a small but non-zero threshold
                if not any(iu.calculate_overlap_percentage(existing_speech_bubble.bounding_box,
                                                           speech_bubble.bounding_box) > 0.7
                           for existing_speech_bubble in best_panel.speech_bubbles):
                    best_panel.speech_bubbles.append(speech_bubble)

        # Sort speech bubbles within each panel
        for panel in page.panels:
            panel.speech_bubbles = sorted(
                panel.speech_bubbles,
                key=lambda p: (p.bounding_box['y'], p.bounding_box['x'])
            )

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
