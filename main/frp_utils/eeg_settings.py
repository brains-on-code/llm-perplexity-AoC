from main.frp_utils.snippet_settings import CONDITION_CLEAN, CONDITION_CONFUSING
from main.frp_utils.visual_settings import *

EEG_FREQUENCY = 500
MNE_KEY_FREQUENCY = 'sfreq'

EOG_CHANNELS = ['VEOGP', 'VEOGM', 'HEOGP', 'HEOGM']
EEG_CHANNELS = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC3', 'FCz', 'FC4', 'FC6',
                'T7', 'C3', 'Cz', 'C4', 'T8', 'CP3', 'CPz', 'CP4', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2']

########
# EEG STIMULI
########
EEG_STIMULUS_SNIPPET_START = {
    CONDITION_CLEAN: 'Stimulus/S 11', CONDITION_CONFUSING: 'Stimulus/S 12'}

#######
# FRP marker
#######
FRP_EEG_STIMULUS_SNIPPET_START = {
    FIXATION_SELECTION_ALGORITHM_COMBINED_PADDING_SMALL_1: {CONDITION_CLEAN: 'Stimulus/S 101',
                                                            CONDITION_CONFUSING: 'Stimulus/S 101'},
    FIXATION_SELECTION_ALGORITHM_COMBINED_PADDING_SMALL_2: {CONDITION_CLEAN: 'Stimulus/S 102',
                                                            CONDITION_CONFUSING: 'Stimulus/S 102'},
    FIXATION_SELECTION_ALGORITHM_COMBINED_PADDING_SMALL_3: {CONDITION_CLEAN: 'Stimulus/S 103',
                                                            CONDITION_CONFUSING: 'Stimulus/S 103'},
    FIXATION_SELECTION_ALGORITHM_COMBINED_PADDING_SMALL_4: {CONDITION_CLEAN: 'Stimulus/S 104',
                                                            CONDITION_CONFUSING: 'Stimulus/S 104'},
    FIXATION_SELECTION_ALGORITHM_COMBINED_PADDING_SMALL_5: {CONDITION_CLEAN: 'Stimulus/S 105',
                                                            CONDITION_CONFUSING: 'Stimulus/S 105'},
    FIXATION_SELECTION_ALGORITHM_COMBINED_PADDING_SMALL_6: {CONDITION_CLEAN: 'Stimulus/S 106',
                                                            CONDITION_CONFUSING: 'Stimulus/S 106'},
    FIXATION_SELECTION_ALGORITHM_COMBINED_PADDING_SMALL_7: {CONDITION_CLEAN: 'Stimulus/S 107',
                                                            CONDITION_CONFUSING: 'Stimulus/S 107'},
    FIXATION_SELECTION_ALGORITHM_COMBINED_PADDING_SMALL_8: {CONDITION_CLEAN: 'Stimulus/S 108',
                                                            CONDITION_CONFUSING: 'Stimulus/S 108'},
    FIXATION_SELECTION_ALGORITHM_COMBINED_PADDING_SMALL_9: {CONDITION_CLEAN: 'Stimulus/S 109',
                                                            CONDITION_CONFUSING: 'Stimulus/S 109'}
}

ERP_CLUSTER_ORIENTATION_INTEGER_CODING = {
    'Fp': 1,
    'F': 3,
    'FC': 4,
    'C': 5,
    'CP': 6,
    'P': 7,
    'O': 9,

    'z': 0,
    '1': -2,  # since only Fp has 1 and 2, and they align with the 3/4 row from other electrodes
    '2': +2,
    '3': -2,
    '4': +2,
    '5': -3,
    '6': +3,
    '7': -4,
    '8': +4}
