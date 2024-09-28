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
                    entities.append(entity)
                    filtered_tags = [tag[0] for tag in entity.tags if tag[1] >= confidence]
                    entity_tags.append(filtered_tags)

            mlb = None
            one_hot_encoded_tags = None
            if input_type in ['One-Hot Encoding', 'TF-IDF-scaled One-Hot', 'TF-IDF-scaled Word2Vec']:
                mlb = MultiLabelBinarizer()
                one_hot_encoded_tags = mlb.fit_transform([tag[0] for tags in entity_tags for tag in tags])

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

