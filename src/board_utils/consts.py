from enum import Enum


class Canny(Enum):
    strong_edge = 255
    weak_edge = 75
    lower_threshold = 50
    upper_threshold = 200
