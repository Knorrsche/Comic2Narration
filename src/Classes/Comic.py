from .Series import Series
from .Page import Page
from typing import Optional, List


class Comic:
    def __init__(self, name: str, volume: int, main_series: Series, secondary_series: Optional['List[Series]'],page_pairs: List[tuple[Optional[Page],Optional[Page]]]):
        self.name = name
        self.volume = volume
        self.main_series = main_series
        self.secondary_series = secondary_series
        self.page_pairs = page_pairs
