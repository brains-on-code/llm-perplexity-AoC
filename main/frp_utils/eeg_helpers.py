from typing import Any

import mne
import pandas as pd
from mne.io import Raw
from mne.io.brainvision.brainvision import RawBrainVision

from main.frp_utils.eeg_settings import *
from main.frp_utils.file_helpers import *
from main.frp_utils.file_settings import *
from main.frp_utils.path_helpers import *
from main.frp_utils.path_settings import *
from main.frp_utils.snippet_helpers import *
from main.frp_utils.snippet_settings import *
from main.frp_utils.textconstants import *
from main.frp_utils.visual_settings import *


def check_file_existence(files: dict[str, Path], file: Path, file_ending: str, participant: str):
    '''check whether file has the given extension and there already exists one.

    Arguments: 
    * files: where to add suitable files per ending
    * file: the path to check
    * file_endings: the file ending to check for
    * participant: the participant to name when problems arise

    raises: Exception if file with given ending has already been identified
    '''
    if file.suffix == file_ending:
        if file_ending in files:
            print(
                f'Multiple \'{file_ending}\' files found for participant {participant}: {files[file_ending].name}, new: {file.name}')
            raise Exception()
        files[file_ending] = file


EEG_HEADER_DATE = re.compile(r'Impedance \[kOhm\] at (\d\d:\d\d:\d\d) :')
EEG_MARKER_DATE = re.compile(r'New Segment,,(\d+),1,0,(\d+)')


def load_eeg_data(filepath: Path, preload: bool = True) -> tuple[None, None] | tuple[RawBrainVision, Any]:
    # check if file exists
    if not filepath.exists():
        print(f'EEG file {filepath} does not exist.')
        return None, None

    raw_eeg = mne.io.read_raw_brainvision(filepath, eog=tuple(EOG_CHANNELS),
                                          preload=preload, verbose=False)
    frequency = raw_eeg.info[MNE_KEY_FREQUENCY]
    return raw_eeg, frequency


def prepare_annotation_information(eeg_data: Raw) -> tuple[pd.DataFrame, float]:
    '''extracts and prepares annotation information from the given EEG data

    Arguments: eeg_data: the eeg data

    returns: a tuple of
    * the annotations as DataFrame with columns onset, duration and description extracted from the eeg data, as well as an additional column describing the onset as a float value to be used for cropping, reinsert, ...
    * the duration of the eeg data in seconds
    '''
    # get recording information required+
    # offset in seconds between start of the file counter and start of samples
    if eeg_data is None:
        print('No EEG data provided.')
        return pd.DataFrame(), 0.0

    recording_offset = eeg_data.first_time
    recording_duration = (eeg_data.n_times - 1) / \
                         eeg_data.info[MNE_KEY_FREQUENCY]  # duration of recording

    annotation_data = eeg_data.annotations.to_data_frame()
    # transform onset of an annotation into a float based on recording start, erasing offset, to use for cropping
    annotation_data[ANNOTATION_COLUMN_ONSET_FLOAT] = annotation_data[ANNOTATION_COLUMN_ONSET].apply(
        lambda onset: (onset - pd.Timestamp(year=1970, month=1, day=1)).total_seconds() - recording_offset)

    return annotation_data, recording_duration


def export_eeg_brainvision(eeg_data: Raw, eeg_path: Path):
    assert (eeg_path.suffix == EEG_FILE_HEADER_ENDING)
    eeg_data.export(eeg_path, overwrite=True, verbose=False)
    eeg_marker_path = eeg_path.with_suffix(EEG_FILE_MARKER_ENDING)
    with open(eeg_marker_path, 'r+') as f:
        eeg_marker_content = f.read()
        f.seek(0)
        eeg_marker_content = eeg_marker_content.replace(
            'Comment,Bad Interval/', 'Bad Interval,')
        eeg_marker_content = eeg_marker_content.replace(
            'Comment,Bad', 'Bad Interval,Bad')
        f.write(eeg_marker_content)
        f.truncate()


def get_all_eeg_trial_segments(erp_frp: bool | str, participant: str, correct_trials: list[str] = None,
                               accepted_snippet_numbers: list[int] = None) -> dict[str, Raw]:
    '''load eeg trial segments for participant

    Arguments:
    * erp_frp: whether it is erp or a certain type of frp
    * participant: the participant to get segments for
    * visual_exclude: whether to exclude based on visual as well (required for FRP)

    returns: the non-excluded trial segments for this participant
    '''
    assert (erp_frp in [True, False]), erp_frp
    modes = [BEHAVIORAL, EEG]
    if not erp_frp:
        modes.append(VISUAL)
    exclusions = get_exclusions(participant, [SNIPPET], modes)[SNIPPET]
    snippet_segments: dict[str, Raw] = {}
    for snippet in exclusions:
        if any(exclusions[snippet].values()):
            # print(f'\t{snippet} was excluded due to {exclusions[snippet]}')
            continue
        # if correct only (correct trials given) and snippet not correctly answered
        if not (correct_trials is None) and not (snippet in correct_trials):
            # print(f'\t{snippet} was excluded due to being answered incorrectly')
            continue
        # if correct only (correct trials given) and snippet not correctly answered
        if not (accepted_snippet_numbers is None) and not (get_snippet_number(snippet) in accepted_snippet_numbers):
            # print(f'\t{snippet} was excluded due to not being in the group')
            continue
        eeg_data, _ = load_eeg_data(get_eeg_trial_path(erp_frp is True, participant, snippet))
        snippet_segments[snippet] = eeg_data
    return snippet_segments


def add_frp_marker_by_special_fixations(participant: str, snippets: list[str], behavioral_data: pd.DataFrame,
                                        special_fixation_data: pd.DataFrame,
                                        eeg_trials: dict[str, mne.io.Raw]) -> None:
    for snippet in snippets:
        snippet_behavioral_start = behavioral_data[behavioral_data[SNIPPET] == snippet].squeeze()[
            BEHAVIORAL_COLUMN_START]
        eeg_segment_data: mne.io.Raw = eeg_trials[snippet]
        snippet_condition = CONDITION_VARIANT_MATCH[get_snippet_variant(snippet)]
        eeg_annotations, _ = prepare_annotation_information(eeg_segment_data)
        if eeg_annotations is None or eeg_annotations.empty:
            print(f'No annotations found for {snippet} in {participant}')
            continue
        eeg_start = eeg_annotations[
            eeg_annotations[ANNOTATION_COLUMN_DESCRIPTION] == EEG_STIMULUS_SNIPPET_START[snippet_condition]].squeeze()[
            ANNOTATION_COLUMN_ONSET_FLOAT]
        snippet_special_fixation_data: pd.DataFrame = special_fixation_data[special_fixation_data[SNIPPET] == snippet]
        for f_a in snippet_special_fixation_data[FIXATION_SELECTION_ALGORITHM].unique():
            fixation_start = \
                snippet_special_fixation_data[
                    snippet_special_fixation_data[FIXATION_SELECTION_ALGORITHM] == f_a].squeeze()[
                    FIXATION_COLUMN_START]
            eeg_fixation_start = transform_eye_to_eeg(fixation_start, snippet_behavioral_start, eeg_start)
            eeg_segment_data.annotations.append(eeg_fixation_start, 1 / EEG_FREQUENCY,
                                                FRP_EEG_STIMULUS_SNIPPET_START[f_a][snippet_condition])
        export_eeg_brainvision(eeg_segment_data, get_eeg_trial_path(False, participant, snippet))


# transform eye-tracking timestamp to eeg frame
def transform_eye_to_eeg(eye_timestamp: float, eye_start_time: float, eeg_start_time: float) -> float:
    '''transforms the eye-tracking timestamp into an eeg frame

    Arguments:
    * eye_timestamp: timestamp of eye-tracking to transform (in seconds with milliseconds floating precision)
    * eye_start_time: timestamp of eye-tracking marking the starting point (in seconds with milliseconds floating precision)
    * eeg_start_time: frame that corresponds to the eye_start timestamp
    * eeg_sampling_rate: the frequency of frames logged in the data (frames per second)

    returns: the frame corresponding to the eye_timestamp
    '''
    eye_offset = eye_timestamp - eye_start_time
    eeg_timestamp = eeg_start_time + eye_offset
    return eeg_timestamp
