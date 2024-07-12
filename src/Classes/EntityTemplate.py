from typing import Dict, List


class EntityTemplate:
    def __init__(self, name: str, images: Dict[str, List[List[List[int]]]]):
        self.name = name
        self.images = images

    def add_entity(self, image: List[List[int]], series: str):
        if series in self.images:
            self.images[series].append(image)
        else:
            self.images[series] = [image]
