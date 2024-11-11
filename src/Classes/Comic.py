import math
import os
from collections import defaultdict

from gensim.models import Word2Vec, KeyedVectors
from sklearn.mixture import GaussianMixture

from .Series import Series
from .Page import Page
from typing import Optional, List
import xml.etree.ElementTree as eT
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer, normalize
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, AffinityPropagation, Birch, DBSCAN
import google.generativeai as genai
import numpy as np
import cv2


class Comic:
    def __init__(self, name: str, volume: int, main_series: Series, secondary_series: Optional['List[Series]'],
                 page_pairs: List[tuple[Optional[Page], Optional[Page]]]):
        self.name = name
        self.volume = volume
        self.main_series = main_series
        self.secondary_series = secondary_series
        self.page_pairs = page_pairs
        self.scenes = []
        self.scene_data = ''

    def to_narrative(self) -> str:
        script = ''
        white_spaces = '     '

        for page_pair in self.page_pairs:

            for page in page_pair:
                if page is None:
                    continue
                script += f'\nPage: {page.page_index}\n'
                panel_counter = 1

                for panel in page.panels:
                    script += f'\n{white_spaces}Panel {panel_counter}: {panel.description}\n'
                    panel_counter += 1
                    speech_bubble_counter = 1

                    for speech_bubble in panel.speech_bubbles:
                        script += f'\n{white_spaces * 2}Speech Bubble {speech_bubble_counter}: {speech_bubble.text}\n'
                        speech_bubble_counter += 1
        return script

    # TODO: create series object
    def to_xml(self):
        element = eT.Element('Comic')
        eT.SubElement(element, 'Name').text = self.name
        eT.SubElement(element, 'Volume').text = str(self.volume)
        # ET.SubElement(element, 'MainSeries').text = self.main_series.name
        eT.SubElement(element, 'MainSeries').text = 'MainSeries'

        secondary_series_element = eT.SubElement(element, 'SecondarySeries')
        # for series in self.secondary_series:
        series_name_element = eT.SubElement(secondary_series_element, 'Series')
        # ET.SubElement(series_name_element, 'Name').text = series.name
        eT.SubElement(series_name_element, 'Name').text = 'SecondarySeries'

        page_pairs_element = eT.SubElement(element, 'PagePairs')
        for pair in self.page_pairs:
            pair_element = eT.SubElement(page_pairs_element, 'PagePair')
            if pair[0] and pair[0] is not None:
                pair_element_left = eT.SubElement(pair_element, 'LeftPage')
                pair_element_left.append(pair[0].to_xml())
            if pair[1] and pair[1] is not None:
                pair_element_right = eT.SubElement(pair_element, 'RightPage')
                pair_element_right.append(pair[1].to_xml())

        return element

    def find_clusters(self):
        model_name: str = "gemini-1.5-flash-002"
        model_gemini = genai.GenerativeModel(model_name=model_name)
        genai.configure(api_key="AIzaSyCoUfTjZU-zNZ2lNKY_BnDuyNTu8lHQ9EM")

        entity_tag_str = "List of Entities: \n"
        entity_counter = 1
        panel_counter = 1
        for scene in self.scenes:
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
            "Calulate a clustersize estimate for following list of enitties: \n")

        prompt += entity_tag_str

        response = model_gemini.generate_content([prompt])

        print(response.text)

    def reset_scenes(self):
        for scene in self.scenes:
            for panel in scene:
                panel.starting_tag = False
                panel.scene_id = 0
        self.scenes = []

    def reset_entities(self):
        for scene in self.scenes:
            for panel in scene:
                panel.entities.clear()

    def resize_image(self,image, target_size):
        return cv2.resize(image, (target_size, target_size))

    def enhanced_matching(self, save_path="output_entities"):
        os.makedirs(save_path, exist_ok=True)  # Create the output directory if it doesn't exist

        for scene_idx, scene in enumerate(self.scenes):
            # Gather all entity images from each panel in the current scene
            entity_images = [entity.image for panel in scene for entity in panel.entities]
            num_entities = len(entity_images)

            if num_entities == 0:
                print(f"No entities found in scene {scene_idx + 1}")
                continue

            # Determine grid size based on the number of entities
            grid_cols = math.ceil(math.sqrt(num_entities))  # Number of columns in the grid
            grid_rows = math.ceil(num_entities / grid_cols)  # Number of rows in the grid

            # Determine maximum width and height of entity images to create a consistent grid
            max_entity_height = max(entity.shape[0] for entity in entity_images)
            max_entity_width = max(entity.shape[1] for entity in entity_images)

            # Create a canvas for arranging entity images in a grid format
            canvas_height = grid_rows * max_entity_height
            canvas_width = grid_cols * max_entity_width
            canvas = np.zeros((canvas_height, canvas_width, 3), dtype=entity_images[0].dtype)

            # Place each entity image onto the canvas
            for idx, entity_img in enumerate(entity_images):
                row = idx // grid_cols
                col = idx % grid_cols
                y_offset = row * max_entity_height
                x_offset = col * max_entity_width

                # Add the entity image to the canvas at the calculated position
                canvas[y_offset:y_offset + entity_img.shape[0], x_offset:x_offset + entity_img.shape[1]] = entity_img

            # Save the combined image for this scene's entities
            file_path = f"{save_path}/scene_{scene_idx + 1}_entities.jpg"
            cv2.imwrite(file_path, canvas)
            print(f"Combined entity image saved at: {file_path}")

    def get_scene_images(self):
        comic_pages = []
        scene_images = []
        for page_pair in self.page_pairs:
            for page in page_pair:
                if page is not None and page not in comic_pages:
                    comic_pages.append(page)

        for idx, scene in enumerate(self.scenes):
            scene_pages = []
            used_pages = []

            for panel in scene:
                if comic_pages[panel.page_id]not in used_pages:
                    page_image = comic_pages[panel.page_id].page_image

                    if page_image.shape[2] == 3:
                        page_image = cv2.cvtColor(page_image, cv2.COLOR_RGB2BGR)

                    page_image = page_image.astype(np.uint8)

                    scene_pages.append(page_image)
                    used_pages.append(comic_pages[panel.page_id])

            if scene_pages:
                try:
                    scene_image = np.hstack(scene_pages)
                    scene_images.append((scene_image,used_pages))
                    #cv2.imwrite(f'scene_{idx}.png', scene_image)
                except ValueError as e:
                    print(f"Error stacking images for scene {idx}: {e}")
        return scene_images




    #TODO: Add temp saving of word2vec to not need to recalculate every time
    def match_entities(self, clusters_list, algorithm='Birch', input_type='One-Hot Encoding', confidence=0.1,
                       debug=False):
        """
        Match entities into clusters based on various clustering algorithms and feature encodings.

        Parameters:
        -----------
        clusters_list : list
            A list containing the number of clusters for each scene.
        algorithm : str, optional
            The clustering algorithm to be used. Choices are ['KMeans', 'DBSCAN', 'Agglomerative', 'Gaussian Mixture', 'Birch'].
            Default is 'Birch'.
        input_type : str, optional
            The data encoding type to be used. Choices are ['One-Hot Encoding', 'TF-IDF-scaled One-Hot', 'Word2Vec', 'TF-IDF-scaled Word2Vec'].
            Default is 'One-Hot Encoding'.
        confidence : float, optional
            Minimum confidence level required for tags to be considered in clustering. Default is 0.1.
        debug : bool, optional
            Whether to print the clusters. Default is False.

        Returns:
        --------
        None
        """
        w2v_model = None
        if input_type in ['Word2Vec', 'TF-IDF-scaled Word2Vec']:
            w2v_model = KeyedVectors.load_word2vec_format(
                r'C:\Users\derra\Downloads\GoogleNews-vectors-negative300.bin', binary=True)

        for counter, scene in enumerate(self.scenes):
            entities = []
            tf_idf = None
            entity_tags = []

            for panel in scene:
                for entity in panel.entities:
                    if not entity.active_tag:
                        continue
                    entities.append(entity)
                    entity.tags = [tag for tag in entity.tags if tag[1] >= confidence]
                    entity_tags.append([tag[0] for tag in entity.tags])

            mlb = None
            one_hot_encoded_tags = None
            if input_type in ['One-Hot Encoding', 'TF-IDF-scaled One-Hot', 'TF-IDF-scaled Word2Vec']:
                mlb = MultiLabelBinarizer()
                one_hot_encoded_tags = mlb.fit_transform(entity_tags)

            tf_idf_vector = None
            if input_type in ['TF-IDF-scaled One-Hot', 'TF-IDF-scaled Word2Vec']:
                tf_idf = self.calculate_tf_idf(scene)
                if input_type == 'TF-IDF-scaled One-Hot' and mlb:
                    tf_idf_vector = np.array([tf_idf.get(tag, 0) for tag in mlb.classes_])

            word2vec_matrix = None
            if input_type in ['Word2Vec', 'TF-IDF-scaled Word2Vec']:
                word2vec_matrix = np.zeros((len(entity_tags), 300))
                for i, tags in enumerate(entity_tags):
                    valid_vectors = [w2v_model[tag[0]] for tag in tags if tag[0] in w2v_model]
                    word2vec_matrix[i] = np.mean(valid_vectors, axis=0) if valid_vectors else np.zeros(300)

            data_matrix = None
            if input_type == 'One-Hot Encoding':
                data_matrix = one_hot_encoded_tags
            elif input_type == 'TF-IDF-scaled One-Hot':
                if tf_idf_vector is not None:
                    data_matrix = one_hot_encoded_tags * tf_idf_vector
            elif input_type == 'Word2Vec':
                data_matrix = word2vec_matrix
            elif input_type == 'TF-IDF-scaled Word2Vec':
                valid_indices = []
                valid_tf_idf_vector = None

                if mlb and tf_idf_vector is not None:
                    valid_tags = mlb.classes_
                    valid_indices = [i for i, tag in enumerate(valid_tags) if tag in w2v_model]
                    valid_tf_idf_vector = tf_idf_vector[valid_indices]

                if valid_tf_idf_vector is not None:
                    valid_word2vec_matrix = word2vec_matrix[:, valid_indices]
                    data_matrix = valid_word2vec_matrix * valid_tf_idf_vector[np.newaxis, :]

            if data_matrix is None or data_matrix.size == 0:
                if debug:
                    print(f"No features found for scene {counter} using {input_type}")
                    print(f"one_hot_encoded_tags: {one_hot_encoded_tags}")
                    print(f"tf_idf_vector: {tf_idf_vector}")
                    print(f"word2vec_matrix: {word2vec_matrix}")
                    print(f"data_matrix: {data_matrix}")
                continue

            if debug:
                print(f"\nScene {counter} - Using {input_type} with shape {data_matrix.shape}:")

            algorithms = {
                'KMeans': KMeans(n_clusters=clusters_list[counter]),
                'DBSCAN': DBSCAN(eps=0.5, min_samples=5),
                'Agglomerative': AgglomerativeClustering(n_clusters=clusters_list[counter]),
                'Gaussian Mixture': GaussianMixture(n_components=clusters_list[counter]),
                'Birch': Birch(n_clusters=clusters_list[counter])
            }

            if algorithm not in algorithms:
                if debug:
                    print(f"Invalid algorithm '{algorithm}' specified.")
                continue

            selected_algorithm = algorithms[algorithm]

            try:
                clusters = selected_algorithm.fit_predict(data_matrix)
                if debug:
                    print(f"{algorithm} Clustering Results: {clusters}")

                for int_, cluster in enumerate(clusters):
                    entities[int_].named_entity_id = cluster

            except Exception as e:
                if debug:
                    print(f"Error running {algorithm} on {input_type}: {e}")




    #TODO: Find better algorithm to identify wrongly classified images
    def update_entities(self, entity_confidence_minimum):
        for scene in self.scenes:
            for panel in scene:
                for entity in panel.entities:
                    has_active_tag = any(tag[1] > entity_confidence_minimum for tag in entity.tags)
                    entity.active_tag = has_active_tag


    #TODO can be more efficent
    def update_scenes(self):
        scene_counter = 0
        scenes = []
        current_scene = []

        for page_pair in self.page_pairs:
            for page in page_pair:
                if not page:
                    continue
                for panel in page.panels:
                    panel.scene_id = scene_counter

                    if scene_counter == 0 and len(current_scene) == 0:
                        current_scene.append(panel)
                        continue

                    if panel.starting_tag:
                        scene_counter += 1
                        panel.scene_id = scene_counter
                        scenes.append(current_scene)

                        current_scene = []

                    current_scene.append(panel)
        if current_scene:
            scenes.append(current_scene)
        self.scenes = scenes

    def calculate_tf(self, scene):
        tag_frequency = defaultdict(int)
        total_tags = 0
        for panel in scene:
            for entity in panel.entities:
                for tag_ in entity.tags:
                    tag = tag_[0]
                    tag_frequency[tag] += 1
                    total_tags += 1
        if total_tags == 0:
            return {}
        tf = {tag: count / total_tags for tag, count in tag_frequency.items()}

        return tag_frequency, tf

    def calculate_idf(self, scene):
        tag_document_count = defaultdict(int)
        total_scenes = len(scene)

        for panel in scene:
            unique_tags = set()
            for entity in panel.entities:
                for tag_ in entity.tags:
                    unique_tags.add(tag_[0])

            for tag in unique_tags:
                tag_document_count[tag] += 1

        idf = {tag: math.log(total_scenes / (1 + count)) for tag, count in tag_document_count.items()}
        return idf

    def calculate_tf_idf(self,scenes):
        tf = self.calculate_tf(scenes)
        idf = self.calculate_idf(scenes)

        tf_idf = {tag: tf_val * idf.get(tag,0) for tag,tf_val in tf[1].items()}
        return tf_idf

