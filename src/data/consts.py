from pathlib import Path

# Paths to directories that contain board images that require parsing
# This is split to different types of boards:
#       - standard position board with all the pieces
#       - board with king and queen with their places swapped (to compensate for them being only on one color in
#           the first type of boards)
DATA_DIR = Path('../../dataset')
DIRS_TO_PARSE_NAMES = ['full_boards', 'replaced_king_queen']
DIRS_TO_PARSE_NAMES = [Path(data_dir) for data_dir in DIRS_TO_PARSE_NAMES]

# Path to the directory to output the parsed boards
OUTPUT_DIR_NAME = 'pieces'
OUTPUT_DIR_PATH = DATA_DIR / OUTPUT_DIR_NAME

# Relevant squares to parse in each type of input board
BOARD_SIDE_SIZE = 8
RELEVANT_SQUARES = {'full_boards': {'rows': [0, 1, -2, -1],
                                    'cols': range(BOARD_SIDE_SIZE)
                                    },
                    'replaced_king_queen': {'rows': [0, -1],
                                            'cols': [BOARD_SIDE_SIZE//2 - 1, BOARD_SIDE_SIZE//2]
                                            }
                    }
