from main.frp_utils.path_settings import *
from main.frp_utils.textconstants import *


def get_screenshot_path(snippet: str, aoi_version: bool = False) -> Path:
    '''creates screenshot path from snippet name and purpose

    Arguments:
    * snippet: the snippet name
    * aoi_version: whether to use the aoi version instead the normal screenshot

    returns: the path of the screenshot, using the aoi_version if specified, and using a dummy link if no file found
    '''
    screenshot_base_path = SCREENSHOTS_PATH_AOI if aoi_version else SCREENSHOTS_PATH_NORMAL
    snippet = snippet.split('.')[0]
    screenshot_path = screenshot_base_path / f'{snippet}.png'
    if screenshot_path.exists():
        return screenshot_path
    else:
        print(
            f'No file for this snippet {snippet} in version \'{"aoi" if aoi_version else "normal"}\' found.')
        return SCREENSHOTS_PATH_DUMMY


def get_behavioral_data_path(participant: str = TOTAL, final_data_exclusion: bool = True,
                             analysis_prep: bool = False) -> Path:
    '''creates behavioral data path from participant marker
    Arguments: participant: participant marker
    returns: the paths of the behavioral event file, using the participant path if specified, otherwise for all participants
    '''
    assert final_data_exclusion if analysis_prep else True, 'If LMER_prep is True, final_data_exclusion must be True.'
    path = PROCESSED_PATH / participant / BEHAVIORAL / (f'{BEHAVIORAL}' +
                                                        ((
                                                             '_analysis' if analysis_prep else '_included') if final_data_exclusion else '') + '.csv')
    path.parent.mkdir(exist_ok=True, parents=True)
    return path


def get_eeg_trial_path(erp_frp: bool, participant: str, snippet: str) -> Path:
    '''creates eeg trial path from participant marker
    Arguments:
    * erp_frp: whether it is erp or a certain type of frp
    * participant: participant marker
    * snippet: the name of the snippet file
    returns: the paths of the eeg trial file, using the participant path if specified
    '''
    assert (erp_frp in [True, False]), erp_frp
    path = PROCESSED_PATH / participant / EEG / (f'{EEG_ERP if erp_frp is True else EEG_FRP}') / \
           f'''{EEG_ERP if erp_frp is True else EEG_FRP}_trial_{
           snippet}{EEG_FILE_HEADER_ENDING}'''
    path.parent.mkdir(exist_ok=True, parents=True)
    return path


def get_exclusion_path(participant: str = TOTAL) -> Path:
    '''creates exclusion file path from participant marker
    Arguments: participant: participant marker
    returns: the path of the exclusion file, using the participant path if specified, otherwise for all participants
    '''
    path = PROCESSED_PATH / participant / 'exclusions.json'
    path.parent.mkdir(exist_ok=True, parents=True)
    return path


def get_selected_fixations_path(participant: str = TOTAL, snippet: str = '') -> Path:
    '''creates fixations path from participant marker
    Arguments: 
    * participant: participant marker
    * snippet: the name of the snippet file
    returns: the paths of the fixations file, using the participant path if specified, otherwise for all participants
    '''
    path = PROCESSED_PATH / participant / VISUAL / \
           FIXATIONS / 'selected' / f'{FIXATIONS}_{snippet}.csv'
    path.parent.mkdir(exist_ok=True, parents=True)
    return path


def get_refixation_plot_path(participant: str, snippet: str) -> Path:
    '''creates special fixations path from participant marker
    Arguments: 
    * participant: participant marker
    returns: the paths of the LMEM model file
    '''
    path = PROCESSED_PATH / participant / VISUAL / \
           FIXATIONS / f'refixation/refixation_plot{snippet}.png'
    path.parent.mkdir(exist_ok=True, parents=True)
    return path


def get_special_fixations_path(participant: str = TOTAL) -> Path:
    '''creates special fixations path from participant marker
    Arguments: 
    * participant: participant marker
    returns: the paths of the LMEM model file
    '''
    path = PROCESSED_PATH / participant / VISUAL / \
           FIXATIONS / ('special_fixations' + '.csv')
    path.parent.mkdir(exist_ok=True, parents=True)
    return path


def get_special_fixation_statistics_path() -> Path:
    '''creates file path for noting all trials where special fixations of a certain algorithm could not be determined
    returns: the file path
    '''
    path = PROCESSED_PATH / TOTAL / VISUAL / \
           ('special_fixations_statistics' + '.csv')
    path.parent.mkdir(exist_ok=True, parents=True)
    return path
