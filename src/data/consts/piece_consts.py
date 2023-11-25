from src.data.consts.path_consts import DIRS_TO_PARSE_NAMES_STR

LABELS = {'P': 0,
          'R': 1,
          'N': 2,
          'B': 3,
          'Q': 4,
          'K': 5,
          'p': 6,
          'r': 7,
          'n': 8,
          'b': 9,
          'q': 10,
          'k': 11,
          'X': 12}

REVERSED_LABELS = {val: key for key, val in LABELS.items()}

BOARD_TO_PIECES_MAP = {
    DIRS_TO_PARSE_NAMES_STR[0]:
        ['rnbqkbnr',
         'pppppppp',
         'XXXXXXXX',
         'XXXXXXXX',
         'XXXXXXXX',
         'XXXXXXXX',
         'PPPPPPPP',
         'RNBQKBNR'],
    DIRS_TO_PARSE_NAMES_STR[1]:
        ['___kq___',
         '________',
         '________',
         '________',
         '________',
         '________',
         '________',
         '___KQ___']
}

PIECE_TO_IGNORE = '_'
