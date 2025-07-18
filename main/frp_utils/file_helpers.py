import json
import re
from pathlib import Path
from typing import Union

from main.frp_utils.path_helpers import get_exclusion_path
from main.frp_utils.path_settings import PROCESSED_PATH, RAW_PATH
from main.frp_utils.textconstants import (BEHAVIORAL, EEG, HDF_FILE, PARTICIPANT,
                           SNIPPET, VISUAL)


def get_participant_folder_per_participant(raw_folder: bool = True) -> dict[str, Path]:
    '''get participant folder per participant identified in the base path
    requirement: the folder must contain three digits in succession, and should not contain any other digits

    returns: the participant numbers, and per each the participant folder
    '''
    folders = {}
    for participant_folder in (RAW_PATH if raw_folder else PROCESSED_PATH).iterdir():
        if not participant_folder.is_dir():
            continue
        # identify participant number
        numbers = re.findall(r'\d\d\d', participant_folder.stem)
        if not numbers:
            print(
                f'in subfolder {participant_folder.as_posix()}, no participant number could be identified.')
            continue
        participant = numbers[0]
        folders[participant] = participant_folder
    return folders


ExclusionLevels = Union[PARTICIPANT, SNIPPET, HDF_FILE]
DataModes = [EEG, BEHAVIORAL, VISUAL]


def get_exclusions(participant: str, levels: list[ExclusionLevels], modes: list[str] = DataModes):
    with open(get_exclusion_path(participant), 'r+') as f:
        exclusions: dict = json.load(f)
        wanted_exclusions = {}
        for level in levels:
            if level == PARTICIPANT:
                wanted_exclusions[PARTICIPANT] = {
                    m: exclusions[PARTICIPANT][m] for m in modes if m in exclusions[PARTICIPANT]}
            else:
                wanted_exclusions[level] = {k: {m: exclusions[level][k][m]
                                                for m in modes if m in exclusions[level][k]} for k in exclusions[level]}
    return wanted_exclusions
