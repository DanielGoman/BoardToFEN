from enum import Enum


class Canny(Enum):
    strong_edge = 255
    weak_edge = 75
    low_thresh_ratio = 0.05
    high_thresh_ratio = 0.15

