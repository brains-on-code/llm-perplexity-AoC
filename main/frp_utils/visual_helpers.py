import math
from typing import Callable

import matplotlib as mpl
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.patches import Circle

from main.frp_utils.aoi_helpers import check_point_in_area, get_distance_to_area
from main.frp_utils.aoi_settings import (AOI_PADDING_SMALL, POS_BOTTOM, POS_LEFT,
                                         POS_RIGHT, POS_TOP)
from main.frp_utils.file_settings import (
    FIXATION_COLUMN_DURATION,
    FIXATION_COLUMN_RADIUS_X,
    FIXATION_COLUMN_RADIUS_Y,
    FIXATION_COLUMN_REFIXATION,
    FIXATION_COLUMN_X,
    FIXATION_COLUMN_Y, FIXATION_COLUMNS,
)
from main.frp_utils.path_helpers import (
    get_refixation_plot_path, get_screenshot_path,
    get_selected_fixations_path,
)
from main.frp_utils.textconstants import OPTIMAL, PARTICIPANT, RESULT, SNIPPET
from main.frp_utils.visual_settings import *


def get_snippet_number(snippet: str) -> str:
    '''get snippet number'''
    assert ('-' in snippet), snippet
    number = snippet.split('-')[0]
    if number == '':
        number = 0
    number = int(number)
    return number


def plot_refixation_clusters(snippet: str, fixation_data: pd.DataFrame, participant: str):
    refixation_clusters = list(
        set(fixation_data[FIXATION_COLUMN_REFIXATION].unique()))
    refixation_clusters.sort()
    if -1 in refixation_clusters:
        refixation_clusters.remove(-1)
    cm_a = mpl.cm.ScalarMappable(
        cmap='hsv', norm=plt.Normalize(vmin=0, vmax=len(refixation_clusters)))
    fig, axis = plt.subplots(1, 1, figsize=(WIDTH / 500, HEIGHT / 500))
    with Image.open(get_screenshot_path(snippet)) as image:
        axis.imshow(image)
        for index, row in fixation_data.iterrows():
            alpha = 0.2
            # color based on index in refixation
            if row[FIXATION_COLUMN_REFIXATION] > -1:
                alpha = 0.2
                c = cm_a.to_rgba(refixation_clusters.index(
                    row[FIXATION_COLUMN_REFIXATION]))
            elif index in refixation_clusters:
                alpha = 0.5
                c = cm_a.to_rgba(refixation_clusters.index(index))
            else:
                c = 'gray'
            p = Circle((row[FIXATION_COLUMN_X], row[FIXATION_COLUMN_Y]),
                       radius=10, color=c, alpha=alpha, linewidth=1)
            axis.add_patch(p)
            axis.axis('off')
    plt.savefig(get_refixation_plot_path(participant, snippet))
    plt.close()


def identify_special_fixations(participant, snippets, aoi_data):
    special_fixation_statistics = pd.DataFrame(
        [], columns=[PARTICIPANT, SNIPPET, FIXATION_SELECTION_ALGORITHM, RESULT])
    special_fixation_data = pd.DataFrame(
        [], columns=FIXATION_COLUMNS + [FIXATION_SELECTION_ALGORITHM])
    for snippet in snippets:
        snippet_fixations_path = get_selected_fixations_path(
            participant, snippet)
        fixation_data = pd.read_csv(
            snippet_fixations_path, index_col=False, sep=';', dtype={PARTICIPANT: str})
        fixation_data = add_refixations(fixation_data)
        plot_refixation_clusters(snippet, fixation_data, participant)
        if snippet in aoi_data[SNIPPET].values:
            aois = aoi_data[aoi_data[SNIPPET] == snippet]
            optimal_aoi = aois[aois[OPTIMAL]]
        else:
            print(f'No aoi found for {snippet}')
            continue
        # use multiple fixation detection algorithms and add markers
        for i, f_a in enumerate(FIXATION_SELECTION_ALGORITHMS):
            algorithm = FIXATION_SELECTION_ALGORITHM_FUNCTIONS[f_a]
            fixation: pd.Series = algorithm(fixation_data, aois)
            if fixation is None:
                print(
                    f'\tFor snippet {snippet}, the algorithm {f_a} could not identify a suitable fixation.')
                special_fixation_statistics.loc[special_fixation_statistics.shape[0]] = {
                    PARTICIPANT: participant, SNIPPET: snippet, FIXATION_SELECTION_ALGORITHM: f_a, RESULT: 'None'}
                continue
            special_fixation_data.loc[special_fixation_data.shape[0]] = (
                    fixation.to_dict() | {FIXATION_SELECTION_ALGORITHM: f_a})
            special_fixation_statistics.loc[special_fixation_statistics.shape[0]] = {
                PARTICIPANT: participant, SNIPPET: snippet, FIXATION_SELECTION_ALGORITHM: f_a, RESULT: 'Existing'}
    return special_fixation_data, special_fixation_statistics


# # Fixations
def add_refixations(fixation_data: pd.DataFrame, ) -> pd.DataFrame:
    '''transforms fixations by adding the refixations column

    A refixation is identified as the first fixation in the sequence (until the fixation to consider) that has less distance than the fixation radius to  the fixation to consider.
    The index of this first fixation is then stored in the new column for the fixation to consider

    Arguments: fixation_data: the fixations located in one participant's view of the snippet

    returns: the modified dataframe
    '''
    fixation_data[FIXATION_COLUMN_REFIXATION] = -1
    for i, row1 in fixation_data.iterrows():
        # look at all previous fixations
        sub_data = fixation_data.loc[:i - 1]
        possible_refixations = {}
        for j, row2 in sub_data.iterrows():
            distance_x = abs(row2[FIXATION_COLUMN_X] - row1[FIXATION_COLUMN_X])
            distance_y = abs(row2[FIXATION_COLUMN_Y] - row1[FIXATION_COLUMN_Y])
            distance = math.sqrt(
                distance_x ** 2 + (REFIXATION_PENALTY_FACTOR_Y * distance_y) ** 2)
            # if the distance is less than the fixation radius, add the previous fixation to the current one's new column
            if distance <= FIXATION_RADIUS:
                possible_refixations[j] = distance
                if row2[FIXATION_COLUMN_REFIXATION] > -1:
                    # refixation transient: refixation on refixated fixation should also be accepted? If so, look for minimum!
                    if row2[FIXATION_COLUMN_REFIXATION] in possible_refixations:
                        possible_refixations[row2[FIXATION_COLUMN_REFIXATION]] = min(
                            possible_refixations[row2[FIXATION_COLUMN_REFIXATION]], distance)
                    else:
                        possible_refixations[row2[FIXATION_COLUMN_REFIXATION]] = distance
        # set refixation to fixation with shortest distance
        if possible_refixations:
            min_arg, min_value = -1, FIXATION_RADIUS
            for index in sorted(possible_refixations.keys()):
                if possible_refixations[index] < min_value - 0.0001:
                    min_arg, min_value = index, possible_refixations[index]
            fixation_data.loc[i, FIXATION_COLUMN_REFIXATION] = min_arg
    return fixation_data


def get_longest_fixation(fixation_data: pd.DataFrame, aoi_data: pd.DataFrame = None,
                         min_duration_s: float = MIN_DURATION_S_FIXATION_EEG) -> pd.Series:
    '''Identifies the longest fixation that extends the minimal duration

    Arguments:
    * fixation_data: the fixations located in one participant's view of the snippet
    * aoi_data: the AoIs identified in the snippet (here unused)
    * min_duration_s: the minimal duration of a fixation to be considered

    returns: the longest fixation longer than min_duration_s if found, else None
    '''
    if fixation_data.empty:
        return None
    longest_fixation_index = fixation_data[FIXATION_COLUMN_DURATION].idxmax()
    longest_fixation = fixation_data.loc[longest_fixation_index]
    if longest_fixation[FIXATION_COLUMN_DURATION] >= min_duration_s:
        return longest_fixation
    return None


def get_longest_refixation(fixation_data: pd.DataFrame, aoi_data: pd.DataFrame = None,
                           min_duration_s: float = MIN_DURATION_S_FIXATION_EEG) -> pd.Series:
    '''Identifies the longest refixation that extends the minimal duration.
    Refixation means that a later fixation has been located at near the location of this fixation.

    Arguments:
    * fixation_data: the fixations located in one participant's view of the snippet
    * aoi_data: the AoIs identified in the snippet (here unused)
    * min_duration_s: the minimal duration of a fixation to be considered

    returns: the longest refixation longer than min_duration_s if found, else None
    '''
    if fixation_data.empty:
        return None
    refixations = fixation_data[fixation_data[FIXATION_COLUMN_REFIXATION] > -1]
    if refixations.empty:
        return None
    longest_refixation_index = refixations[FIXATION_COLUMN_DURATION].idxmax()
    longest_refixation = fixation_data.loc[longest_refixation_index]
    if longest_refixation[FIXATION_COLUMN_DURATION] >= min_duration_s:
        return longest_refixation
    return None


def get_first_fixation_of_most_fixated(fixation_data: pd.DataFrame, aoi_data: pd.DataFrame = None,
                                       min_duration_s: float = MIN_DURATION_S_FIXATION_EEG) -> pd.Series:
    '''Identifies the set of fixations where there is the most refixation, and longest fixation that extends the minimal duration

    Arguments:
    * fixation_data: the fixations located in one participant's view of the snippet
    * aoi_data: the AoIs identified in the snippet (here unused)
    * min_duration_s: the minimal duration of a fixation to be considered as result

    returns: the first fixation with the most other refixations longer than min_duration_s if found, else None
    '''
    fixation_data = fixation_data.reset_index(drop=True)
    fixation_counter = {}
    for i, row in fixation_data.iterrows():
        original_fixation_index = row[FIXATION_COLUMN_REFIXATION]
        # store all refixations in a dictionary by inserting them to their original fixation
        if original_fixation_index > -1:
            if not original_fixation_index in fixation_counter:
                fixation_counter[original_fixation_index] = {
                    'all_count': 0, 'long_count': 0, 'short_count': 0,
                    'list': [original_fixation_index]}
            fixation_counter[original_fixation_index]['all_count'] += 1
            fixation_counter[original_fixation_index]['list'].append(i)
            if row[FIXATION_COLUMN_DURATION] > min_duration_s:
                fixation_counter[original_fixation_index]['long_count'] += 1
            else:
                fixation_counter[original_fixation_index]['short_count'] += 1
    if len(fixation_counter) == 0:
        return None
    # sort the refixation clusters by frequency using comparator
    sorted_fixation_counter = sorted(
        fixation_counter.items(), key=lambda x: (x[1]['all_count'], x[1]['long_count']), reverse=True)
    # for most frequent refixation, take the first that exceeds the minimal duration
    for fixation_index in sorted_fixation_counter[0][1]['list']:
        fixation = fixation_data.loc[fixation_index]
        if fixation[FIXATION_COLUMN_DURATION] >= min_duration_s:
            return fixation
    print('For most refixations no fixation with given duration found.')
    return None


def get_longest_fixation_optimal_aoi(fixation_data: pd.DataFrame, aoi_data: pd.DataFrame, aoi_padding: dict[str, float],
                                     min_duration_s: float = MIN_DURATION_S_FIXATION_EEG) -> pd.Series:
    '''Identifies the longest fixation that extends the minimal duration

    Arguments:
    * fixation_data: the fixations located in one participant's view of the snippet
    * aoi_data: the AoIs identified in the snippet (here unused)
    * min_duration_s: the minimal duration of a fixation to be considered

    returns: the longest fixation longer than min_duration_s if found, else None
    '''
    return get_longest_fixation(get_fixations_in_aoi(fixation_data, aoi_data, aoi_padding), aoi_data, min_duration_s)


def get_longest_refixation_optimal_aoi(fixation_data: pd.DataFrame, aoi_data: pd.DataFrame,
                                       aoi_padding: dict[str, float],
                                       min_duration_s: float = MIN_DURATION_S_FIXATION_EEG) -> pd.Series:
    '''Identifies the longest fixation that extends the minimal duration

    Arguments:
    * fixation_data: the fixations located in one participant's view of the snippet
    * aoi_data: the AoIs identified in the snippet (here unused)
    * min_duration_s: the minimal duration of a fixation to be considered

    returns: the longest fixation longer than min_duration_s if found, else None
    '''
    return get_longest_refixation(get_fixations_in_aoi(fixation_data, aoi_data, aoi_padding), aoi_data, min_duration_s)


def get_first_fixation_optimal_aoi(fixation_data: pd.DataFrame, aoi_data: pd.DataFrame, aoi_padding: dict[str, float],
                                   min_duration_s: float = MIN_DURATION_S_FIXATION_EEG) -> pd.Series:
    '''Identifies the set of fixations where there is the most refixation, and longest fixation that extends the minimal duration

    Arguments:
    * fixation_data: the fixations located in one participant's view of the snippet
    * aoi_data: the AoIs identified in the snippet (here unused)
    * min_duration_s: the minimal duration of a fixation to be considered as result

    returns: the first fixation with the most other refixations longer than min_duration_s if found, else None
    '''
    fixation_data = get_fixations_in_aoi(fixation_data, aoi_data, aoi_padding)
    if fixation_data is None or fixation_data.empty:
        return None
    for i, fixation in fixation_data.iterrows():
        if fixation[FIXATION_COLUMN_DURATION] >= min_duration_s:
            return fixation
    return None


def get_first_fixation(fixation_data: pd.DataFrame, aoi_data: pd.DataFrame,
                       min_duration_s: float = MIN_DURATION_S_FIXATION_EEG) -> pd.Series:
    '''Identifies the first fixation

    Arguments:
    * fixation_data: the fixations located in one participant's view of the snippet
    * aoi_data: the AoIs identified in the snippet (here unused)
    * min_duration_s: the minimal duration of a fixation to be considered as result

    returns: the first fixation with the most other refixations longer than min_duration_s if found, else None
    '''
    return fixation_data.iloc[0]


def get_fixations_in_aoi(fixation_data: pd.DataFrame, aoi_data: pd.DataFrame, aoi_padding: dict[str, int]):
    if aoi_data is None or (aoi_data.empty):
        print('No AOIs given')
        return None
    optimal_aoi = aoi_data[aoi_data[OPTIMAL]].squeeze()
    aoi_area = {POS_BOTTOM: optimal_aoi[POS_BOTTOM] + aoi_padding[POS_BOTTOM],
                POS_TOP: optimal_aoi[POS_TOP] - aoi_padding[POS_TOP],
                POS_LEFT: optimal_aoi[POS_LEFT] - aoi_padding[POS_LEFT],
                POS_RIGHT: optimal_aoi[POS_RIGHT] + aoi_padding[POS_RIGHT]}
    fixation_data = fixation_data[fixation_data.apply(lambda row: check_point_in_area(
        row[FIXATION_COLUMN_X], row[FIXATION_COLUMN_Y], row[FIXATION_COLUMN_RADIUS_X], row[FIXATION_COLUMN_RADIUS_Y],
        aoi_area), axis=1)]
    return fixation_data


def identify_closest_long_fixation(fixation_data: pd.DataFrame, aoi_data: pd.DataFrame,
                                   aoi_padding: dict[str, int], min_duration_s: float = MIN_DURATION_S_FIXATION_EEG,
                                   try_for_better_order: int = 2, try_for_better_distance_worsening: int = 10,
                                   try_for_better_distance_optimizing: int = 5):
    '''identifies the suitable fixation for FRP via duration, distance to AOI and order

    using a sophisticated algorithm:
      - from all fixations longer than a certain duration 
      - get the first fixation within a given boundary of the AOI (use direct AOI boundary retrieved from image)
      - with a certain tradeoff between proximity to AOI and order of fixations

    Arguments:
    * fixation_data: contains the data of the fixations for one participant's trial
    * AOIs: the markers of the AOI borders (in pixels), each given by AOI name and inside given via the 4 position markers '{POS_TOP}', '{POS_LEFT}', '{POS_BOTTOM}', '{POS_RIGHT}'
    * min_duration_s: the minimum required duration of a fixation to be considered for FRP
    * aoi_padding: the accepted padding around an AOI in which a suitable fixation can be considered for FRP (in pixels)
    * try_for_better_order: the amount of fixations with a given duration that shall be tested for better fit after identifying the first suitable fixation
    * try_for_better_distance_worsening: the maximum worsening in distance allowed for finding a better fixation after the first suitable one has been found, considers also unsuitable fixations. (in pixels)
    * try_for_better_distance_optimizing: the distance a better fixation has to reduce after the first suitable one has been found (only for suitable fixations). (in pixels)

    returns: the starting point of a fixation that should be used for FRP, if such could be identified. Otherwise -1
'''
    if aoi_data is None or (aoi_data.empty):
        print('No AOIs given')
        return None
    optimal_aoi = aoi_data[aoi_data[OPTIMAL]].squeeze()
    aoi_area = {POS_BOTTOM: optimal_aoi[POS_BOTTOM] + aoi_padding[POS_BOTTOM],
                POS_TOP: optimal_aoi[POS_TOP] - aoi_padding[POS_TOP],
                POS_LEFT: optimal_aoi[POS_LEFT] - aoi_padding[POS_LEFT],
                POS_RIGHT: optimal_aoi[POS_RIGHT] + aoi_padding[POS_RIGHT]}
    found_better = -1
    for i, row in fixation_data.iterrows():
        # for i, duration_suitable, start_time, x, y, range_x, range_y in fixations:
        x, y = row[FIXATION_COLUMN_X], row[FIXATION_COLUMN_Y]
        radius_x, radius_y = row[FIXATION_COLUMN_RADIUS_X], row[FIXATION_COLUMN_RADIUS_Y]
        if found_better > -1:
            if i < found_better:
                continue
            elif i == found_better:
                found_better = -1
            else:
                print(
                    f'How did this happen: i {i} found_better {found_better}')
                break
        # check that fixation duration is acceptable
        if row[FIXATION_COLUMN_DURATION] > min_duration_s:
            # check that fixation is in acceptable distance to AOI
            if not check_point_in_area(x, y, radius_x, radius_y, aoi_area):
                continue
            # print(
            #     f'found suitable fixation at index {i} and position ({x},{y}). Now check that there is not a better one later on')
            distance_to_AOI, direction_from_AOI = get_distance_to_area(
                x, y, radius_x, radius_y, optimal_aoi)
            # found suitable fixation at index i. Now check that there is not a better one later on
            for i_try, row_try in fixation_data.loc[i + 1:i + try_for_better_order + 1].iterrows():
                x_try, y_try = row_try[FIXATION_COLUMN_X], row_try[FIXATION_COLUMN_Y]
                radius_x_try, radius_y_try = row_try[FIXATION_COLUMN_RADIUS_X], row_try[FIXATION_COLUMN_RADIUS_Y]
                # check tradeoff between this and later fixations
                # check that distance hasn't worsened too much
                distance_to_AOI_try, direction_from_AOI_try = get_distance_to_area(
                    x_try, y_try, radius_x_try, radius_y_try, optimal_aoi)
                if distance_to_AOI_try > try_for_better_distance_worsening + distance_to_AOI:
                    break
                # check that fixation duration is acceptable
                if row_try[FIXATION_COLUMN_DURATION] <= min_duration_s:
                    continue
                # check that fixation is in acceptable distance to AOI
                if not check_point_in_area(x_try, y_try, radius_x_try, radius_y_try, aoi_area):
                    continue
                # check that fixation has a better distance
                if distance_to_AOI_try < distance_to_AOI - try_for_better_distance_optimizing:
                    found_better = i_try
            if found_better == -1:
                return row
    return None


# Mapping of fixation algorithm to function (functions have all identical fixation_data: pd.Series and AOIs, the rest is optional)
FIXATION_SELECTION_ALGORITHM_FUNCTIONS: \
    dict[str, Callable[[pd.DataFrame, pd.DataFrame, float], pd.Series]] = {
    FIXATION_SELECTION_ALGORITHM_COMBINED_PADDING_SMALL_1: lambda fixation_data,
                                                                  aoi_data: identify_closest_long_fixation(
        fixation_data, aoi_data[aoi_data["NO"] == 1], AOI_PADDING_SMALL),
    FIXATION_SELECTION_ALGORITHM_COMBINED_PADDING_SMALL_2: lambda fixation_data,
                                                                  aoi_data: identify_closest_long_fixation(
        fixation_data, aoi_data[aoi_data["NO"] == 2], AOI_PADDING_SMALL),
    FIXATION_SELECTION_ALGORITHM_COMBINED_PADDING_SMALL_3: lambda fixation_data,
                                                                  aoi_data: identify_closest_long_fixation(
        fixation_data, aoi_data[aoi_data["NO"] == 3], AOI_PADDING_SMALL),
    FIXATION_SELECTION_ALGORITHM_COMBINED_PADDING_SMALL_4: lambda fixation_data,
                                                                  aoi_data: identify_closest_long_fixation(
        fixation_data, aoi_data[aoi_data["NO"] == 4], AOI_PADDING_SMALL),
    FIXATION_SELECTION_ALGORITHM_COMBINED_PADDING_SMALL_5: lambda fixation_data,
                                                                  aoi_data: identify_closest_long_fixation(
        fixation_data, aoi_data[aoi_data["NO"] == 5], AOI_PADDING_SMALL),
    FIXATION_SELECTION_ALGORITHM_COMBINED_PADDING_SMALL_6: lambda fixation_data,
                                                                  aoi_data: identify_closest_long_fixation(
        fixation_data, aoi_data[aoi_data["NO"] == 6], AOI_PADDING_SMALL),
    FIXATION_SELECTION_ALGORITHM_COMBINED_PADDING_SMALL_7: lambda fixation_data,
                                                                  aoi_data: identify_closest_long_fixation(
        fixation_data, aoi_data[aoi_data["NO"] == 7], AOI_PADDING_SMALL),
    FIXATION_SELECTION_ALGORITHM_COMBINED_PADDING_SMALL_8: lambda fixation_data,
                                                                  aoi_data: identify_closest_long_fixation(
        fixation_data, aoi_data[aoi_data["NO"] == 8], AOI_PADDING_SMALL),
    FIXATION_SELECTION_ALGORITHM_COMBINED_PADDING_SMALL_9: lambda fixation_data,
                                                                  aoi_data: identify_closest_long_fixation(
        fixation_data, aoi_data[aoi_data["NO"] == 9], AOI_PADDING_SMALL),
}
