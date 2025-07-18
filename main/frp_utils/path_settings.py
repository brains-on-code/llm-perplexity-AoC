from pathlib import Path

from main.frp_utils.textconstants import TOTAL, VISUAL

TRIALS_PATH = Path('')  # Path to the trials data (08-Data-Trial_Recordings)
RAW_PATH = TRIALS_PATH / 'raw'
PROCESSED_PATH = TRIALS_PATH / 'processed'
X_OFFSET_FILE = PROCESSED_PATH / TOTAL / VISUAL / 'accuracy_offset_x_axis.json'

EEG_FILE_DATA_ENDING = '.eeg'
EEG_FILE_HEADER_ENDING = '.vhdr'
EEG_FILE_MARKER_ENDING = '.vmrk'

# Path to screenshot files
SCREENSHOTS_PATH = (TRIALS_PATH / 'screenshots')
SCREENSHOTS_PATH_AOI = SCREENSHOTS_PATH / 'aoi'
SCREENSHOTS_PATH_NORMAL = SCREENSHOTS_PATH / 'normal'
SCREENSHOTS_PATH_DUMMY = SCREENSHOTS_PATH / 'dummy.png'

(SCREENSHOTS_PATH / 'analysis').mkdir(parents=True, exist_ok=True)
