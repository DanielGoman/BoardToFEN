from pathlib import Path

# Paths to directories that contain board images that require parsing
# This is split to different types of boards:
#       - standard position board with all the squares
#       - board with king and queen with their places swapped (to compensate for them being only on one color in
#           the first type of boards)
DATA_DIR = Path('../../dataset')
DIRS_TO_PARSE_NAMES_STR = ['full_boards', 'replaced_king_queen']
DIRS_TO_PARSE_NAMES = [Path(data_dir) for data_dir in DIRS_TO_PARSE_NAMES_STR]

# Path to the directory to output the parsed boards
PIECES_OUTPUT_DIR_NAME = 'squares'
LABELS_OUTPUT_DIR_NAME = 'labels'
PIECES_OUTPUT_DIR_PATH = DATA_DIR / PIECES_OUTPUT_DIR_NAME
LABELS_OUTPUT_DIR_PATH = DATA_DIR / LABELS_OUTPUT_DIR_NAME

LABELS_OUTPUT_FILE_NAME = 'labels.json'
TRAIN_LABELS_OUTPUT_FILE_NAME = 'train_labels.json'
VAL_LABELS_OUTPUT_FILE_NAME = 'val_labels.json'
LABELS_OUTPUT_FILE_PATH = LABELS_OUTPUT_DIR_PATH / LABELS_OUTPUT_FILE_NAME
TRAIN_LABELS_OUTPUT_FILE_PATH = str(LABELS_OUTPUT_DIR_PATH / TRAIN_LABELS_OUTPUT_FILE_NAME)
VAL_LABELS_OUTPUT_FILE_PATH = str(LABELS_OUTPUT_DIR_PATH / VAL_LABELS_OUTPUT_FILE_NAME)


# Creating missing dirs if they do not exist
DATA_DIR.mkdir(exist_ok=True)
for dir_name in DIRS_TO_PARSE_NAMES:
    (DATA_DIR / dir_name).mkdir(exist_ok=True)

PIECES_OUTPUT_DIR_PATH.mkdir(exist_ok=True)
LABELS_OUTPUT_DIR_PATH.mkdir(exist_ok=True)
