
NON_PIECE = 'X'

PIECE_TYPE = {'P': 0,
              'R': 1,
              'N': 2,
              'B': 3,
              'Q': 4,
              'K': 5,
              NON_PIECE: 6}


PIECE_COLOR = {'W': 0,
               'B': 1,
               NON_PIECE: 2}

REVERSED_PIECE_TYPE = {val: key for key, val in PIECE_TYPE.items()}
REVERSED_PIECE_COLOR = {val: key for key, val in PIECE_COLOR.items() if key != NON_PIECE}

