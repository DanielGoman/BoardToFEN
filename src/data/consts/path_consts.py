from pathlib import Path

# Paths to directories that contain board images that require parsing
# This is split to different types of boards:
#       - standard position board with all the squares
#       - board with king and queen with their places swapped (to compensate for them being only on one color in
#           the first type of boards)
DATA_DIR = Path('../../dataset')
DIRS_TO_PARSE_NAMES = ['full_boards', 'replaced_king_queen']
DIRS_TO_PARSE_NAMES = [Path(data_dir) for data_dir in DIRS_TO_PARSE_NAMES]

# Path to the directory to output the parsed boards
PIECES_OUTPUT_DIR_NAME = 'squares'
LABELS_OUTPUT_DIR_NAME = 'labels'
PIECES_OUTPUT_DIR_PATH = DATA_DIR / PIECES_OUTPUT_DIR_NAME
LABELS_OUTPUT_DIR_PATH = DATA_DIR / LABELS_OUTPUT_DIR_NAME

LABELS_OUTPUT_FILE_NAME = 'labels.json'
LABELS_OUTPUT_FILE_PATH = LABELS_OUTPUT_DIR_PATH / LABELS_OUTPUT_FILE_NAME


# Creating missing dirs if they do not exist
DATA_DIR.mkdir(exist_ok=True)
for dir_name in DIRS_TO_PARSE_NAMES:
    (DATA_DIR / dir_name).mkdir(exist_ok=True)

PIECES_OUTPUT_DIR_PATH.mkdir(exist_ok=True)
LABELS_OUTPUT_DIR_PATH.mkdir(exist_ok=True)
