from .Series import Series
from .Page import Page
from typing import Optional, List


class Comic:
    def __init__(self, name: str, volume: int, main_series: Series, secondary_series: Optional['List[Series]'],pages: List[Page]):
        self.name = name
        self.volume = volume
        self.mainSeries = main_series
        self.secondarySeries = secondary_series
        self.pages = pages
    