import math
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


class Comic:
    def __init__(self, name: str, volume: int, main_series: Series, secondary_series: Optional['List[Series]'],
                 page_pairs: List[tuple[Optional[Page], Optional[Page]]]):
        self.name = name
        self.volume = volume
        self.main_series = main_series
        self.secondary_series = secondary_series
        self.page_pairs = page_pairs
        self.scenes = []

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

    def match_entities(self, clusters_list):
        w2v_model = KeyedVectors.load_word2vec_format(r'C:\Users\derra\Downloads\GoogleNews-vectors-negative300.bin',
                                                      binary=True)

        for counter, scene in enumerate(self.scenes):
            entities = []
            tf_idf = self.calculate_tf_idf(scene)

            entity_tags = []
            for panel in scene:
                for entity in panel.entities:
                    entities.append(entity)
                    entity_tags.append([tag_[0] for tag_ in entity.tags])

            mlb = MultiLabelBinarizer()
            one_hot_encoded_tags = mlb.fit_transform(entity_tags)

            tf_idf_vector = np.array([tf_idf.get(tag, 0) for tag in mlb.classes_])

            word2vec_matrix = np.zeros((len(entity_tags), 300))
            for i, tags in enumerate(entity_tags):
                valid_vectors = [w2v_model[tag] for tag in tags if tag in w2v_model]
                word2vec_matrix[i] = np.mean(valid_vectors, axis=0) if valid_vectors else np.zeros(300)

            valid_tags = mlb.classes_
            valid_indices = [i for i, tag in enumerate(valid_tags) if tag in w2v_model]

            valid_tf_idf_vector = tf_idf_vector[valid_indices]
            valid_word2vec_matrix = word2vec_matrix[:, valid_indices]

            tf_idf_scaled_word2vec = valid_word2vec_matrix * valid_tf_idf_vector[np.newaxis, :]  # Correct broadcasting

            data_matrices = {
                'One-Hot Encoding': one_hot_encoded_tags,
                'TF-IDF-scaled One-Hot': one_hot_encoded_tags * tf_idf_vector,  # Ensure this matches the correct shape
                'Word2Vec': word2vec_matrix,
                'TF-IDF-scaled Word2Vec': tf_idf_scaled_word2vec
            }

            if not any(matrix.size for matrix in data_matrices.values()):
                print(f"No features found for scene {counter}")
                continue

            print(f"\nScene {counter} - Running multiple clustering algorithms and encodings:")

            algorithms = {
                'KMeans': KMeans(n_clusters=clusters_list[counter]),
                'DBSCAN': DBSCAN(eps=0.5, min_samples=5),
                'Agglomerative': AgglomerativeClustering(n_clusters=clusters_list[counter]),
                'Gaussian Mixture': GaussianMixture(n_components=clusters_list[counter]),
                'Birch': Birch(n_clusters=clusters_list[counter])
            }

            for matrix_name, matrix in data_matrices.items():
                if matrix.size == 0:
                    continue

                print(f"\nUsing {matrix_name} with shape {matrix.shape}:")

                for algo_name, algorithm in algorithms.items():
                    try:
                        clusters = algorithm.fit_predict(matrix)
                        print(f"{algo_name} Clustering Results: {clusters}")

                        for int_, cluster in enumerate(clusters):
                            entities[int_].named_entity_id = cluster

                    except Exception as e:
                        print(f"Error running {algo_name} on {matrix_name}: {e}")

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

