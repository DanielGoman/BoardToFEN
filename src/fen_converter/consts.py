from enum import Enum


ACTIVE_COLOR_MAPPING = {False: 'w',
                        True: 'b'}

CASTLING_RIGHTS_MAPPING = {0: 'K',
                           1: 'Q',
                           2: 'k',
                           3: 'q'}


class FenPartDefaults(Enum):
    str = '-'
    int = 0
