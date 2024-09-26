from .Series import Series
from .Page import Page
from typing import Optional, List
import xml.etree.ElementTree as eT
from gensim.downloader import api
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


from sklearn.preprocessing import MultiLabelBinarizer, normalize
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, AffinityPropagation, Birch


class Comic:
    def __init__(self, name: str, volume: int, main_series: Series, secondary_series: Optional['List[Series]'],
                 page_pairs: List[tuple[Optional[Page], Optional[Page]]]):
        self.name = name
        self.volume = volume
        self.main_series = main_series
        self.secondary_series = secondary_series
        self.page_pairs = page_pairs
        self.scenes = []
        self.word2vec_model = api.load('word2vec-google-news-300')

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

    # TODO: Change to extern file later
    # add option to enter from - to for panels for the scene
    # and add a cluster input field
    def match_entities(self, clusters_list):
        character_tags = []
        entities = []
        for counter, scene in enumerate(self.scenes):
            for panel in scene:
                for entity in panel.entities:
                    if entity.tags:
                        tags = [item[0] for item in entity.tags]
                        character_tags.append(tags)
                        entities.append(entity)

            mlb_char_tags = MultiLabelBinarizer()
            character_tag_features = mlb_char_tags.fit_transform(character_tags)
            normalized_character_tag_features = normalize(character_tag_features)
            clustering_algorithm = Birch(n_clusters=clusters_list[counter])
            clusters = clustering_algorithm.fit_predict(normalized_character_tag_features)
            for int_, cluster in enumerate(clusters):
                entities[int_].named_entity_id = clusters[int_]
            character_tags = []
            entities = []

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

    def calculate_inv_idf(self):
        inter = 2
