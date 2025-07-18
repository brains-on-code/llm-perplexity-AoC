# AOIs
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

from main.frp_utils.aoi_settings import *
from main.frp_utils.file_settings import *
from main.frp_utils.snippet_helpers import *
from main.frp_utils.snippet_settings import *
from main.frp_utils.textconstants import *


#############################
# AOI detection & retrieval from (screenshot) images
#############################


# ##
# * get all pixels in the colors of the AOIs from the image (in binary view)
# * detect contours


def get_AOI_size_per_snippet(all_aois: dict[str, dict[str, dict[str, float]]]) -> pd.DataFrame:
    '''Create a dataframe with AOI size per snippet and AOI level, treating each snippet individually.'''
    level_area_data = pd.DataFrame(
        columns=[CONDITION, AOI_NO, SNIPPET, f'{SNIPPET}_base', SNIPPET_COLUMN_NUMBER, SNIPPET_COLUMN_VERSION,
                 AOI_SIZE, OPTIMAL, POS_BOTTOM, POS_LEFT, POS_RIGHT, POS_TOP])

    for snippet in tqdm(all_aois):
        snippet_base = get_snippet_base(snippet)
        if SNIPPET_OBF in snippet:
            condition = CONDITION_VARIANT_MATCH[SNIPPET_OBF]  # ambiguous
        elif SNIPPET_CLEAN in snippet:
            condition = CONDITION_VARIANT_MATCH[SNIPPET_CLEAN]  # unambiguous
        else:
            condition = ''
        base_entry = {
            SNIPPET: snippet,
            CONDITION: condition,
            f'{SNIPPET}_base': snippet_base,
            SNIPPET_COLUMN_VERSION: get_snippet_version(snippet),
            SNIPPET_COLUMN_NUMBER: get_snippet_number(snippet),
        }
        for aoi_level, aoi in all_aois[snippet].items():
            level_area_data.loc[len(level_area_data)] = {
                                                            AOI_NO: aoi_level,
                                                            AOI_SIZE: abs(aoi[POS_TOP] - aoi[POS_BOTTOM]) * abs(
                                                                aoi[POS_LEFT] - aoi[POS_RIGHT]),
                                                            OPTIMAL: True
                                                        } | base_entry | aoi
    return level_area_data


def check_point_in_area(x: float, y: float, radius_x: float, radius_y: float, area: dict[str, int]) -> bool:
    f'''checks whether the point or the surrounding range can be found inside the area

    Arguments:
    * x, y: the position of the point to check
    * radius_x, radius_y: the area around the point counting as one (given as diameter in form of a rectangle)
    * area: the area to check for, given via the 4 position markers '{POS_TOP}', '{POS_LEFT}', '{POS_BOTTOM}', '{POS_RIGHT}'

    returns: whether the point (including range) is (partly) inside the area
    '''
    left_x_border, right_x_border = x - radius_x, x + radius_x
    top_y_border, bottom_y_border = y - radius_y, y + radius_y
    if right_x_border < area[POS_LEFT]:
        return False
    if left_x_border > area[POS_RIGHT]:
        return False
    if top_y_border > area[POS_BOTTOM]:
        return False
    if bottom_y_border < area[POS_TOP]:
        return False
    # print(f'Area ({x},{y}) with ranges {range_x}, {range_y} in area {area}')
    return True


def get_distance_to_area(x: float, y: float, range_x: float, range_y: float, area: dict[str, int]) -> float:
    f'''checks whether the point or the surrounding range can be found inside the area

    Arguments:
    * x, y: the position of the point to check
    * range_x, range_y: the area around the point counting as one (given as diameter in form of a rectangle)
    * area: the area to check for, given via the 4 position markers '{POS_TOP}', '{POS_LEFT}', '{POS_BOTTOM}', '{POS_RIGHT}'

    returns:
    * the minimal Euclidean distance of the point (including range) to the area
    * a list of direction markers (0-2) that describe the relative position of the point with regard to the area ('{POS_TOP}' means that the point is above the area)
    '''
    direction_markers = []
    left_x_border, right_x_border = x - range_x / 2, x + range_x / 2
    top_y_border, bottom_y_border = y - range_y / 2, y + range_y / 2
    left_distance = max(0, area[POS_LEFT] - right_x_border)
    right_distance = max(0, left_x_border - area[POS_RIGHT])
    assert ((right_distance == 0) or (left_distance == 0))
    if right_distance > 0:
        direction_markers.append(POS_RIGHT)
    elif left_distance > 0:
        direction_markers.append(POS_LEFT)
    top_distance = max(0, area[POS_TOP] - bottom_y_border)
    bottom_distance = max(0, top_y_border - area[POS_BOTTOM])
    assert ((top_distance == 0) or (bottom_distance == 0))
    if top_distance > 0:
        direction_markers.append(POS_TOP)
    elif bottom_distance > 0:
        direction_markers.append(POS_BOTTOM)
    x_distance = max(left_distance, right_distance)
    y_distance = max(top_distance, bottom_distance)
    return np.linalg.norm([x_distance, y_distance]), direction_markers
