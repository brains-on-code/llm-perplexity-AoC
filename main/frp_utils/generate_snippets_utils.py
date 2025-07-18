import json
import os
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pygments import highlight
from pygments.formatters.bbcode import BBCodeFormatter
from pygments.lexers import get_lexer_by_name
from pygments.styles import get_style_by_name


def parse_bbcode_list(bbcode, logging):
    """parse code letter by letter, removing metadata characters but assigning them to the parsed letter dictionary"""
    result = []
    pattern = r"\[[^\[\]]*?\]"
    idx = 0
    base_color = (0, 0, 0, 255)
    color = base_color
    bold = False
    italic = False
    aoi_default = "None"
    aoi_name = [aoi_default]
    if logging:
        print('bbcode', len(bbcode), bbcode)
    while idx < len(bbcode):
        data = re.search(pattern, bbcode[idx:])
        meta = False

        if data is not None and data.start(0) == 0:
            value = data.group(0).strip("[").strip("]")
            if logging:
                print('value', value)
            if "/aoi" in value:
                meta = True
                aoi_name.pop()
                if logging:
                    print(f"after deletion {aoi_name}")
            elif "aoi" in value:
                meta = True
                aoi_name.append(value[4:])
                if logging:
                    print(f"after insertion {aoi_name}")
            if "/color" in value:
                meta = True
                color = base_color
            elif "color" in value:
                meta = True
                color = tuple(int(value[i: i + 2], 16) for i in (7, 9, 11))
                color = (color[0], color[1], color[2], 255)

            if "/b" == value:
                meta = True
                bold = False
            elif "b" == value:
                meta = True
                bold = True

            #            if "/i" == value:
            #                meta = True
            #                italic = False
            #            elif "i" == value:
            #                meta = True
            #                italic = True

            if "/u" == value:
                meta = True
            elif "u" == value:
                meta = True

            if meta:
                idx += data.end(0)

        if not meta:
            result.append({"letter": bbcode[idx], "color": color, "bold": bold, "italic": italic, "AOI": [
                value for value in aoi_name]})
            idx += 1
    if len(aoi_name) != 1:
        raise Exception("AOI not closed - " + aoi_name[-1])
    return result


def set_font(data, regular, bold, italic, bolditalic):
    """determine font with given font formatting for each letter in the parsed dataset"""
    new_result = []
    for entry in data:
        if entry["bold"] and entry["italic"]:
            entry["font"] = bolditalic
        elif entry["bold"]:
            entry["font"] = bold
        elif entry["italic"]:
            entry["font"] = italic
        else:
            entry["font"] = regular
        new_result.append(entry)

    return new_result


def create_image(source_json, font_path="./fonts/ttf/", logging=False, track_coordinates=False):
    """creates image from code snippet with syntax highlighting and AOIs with the information stored in data, such as background color, width, height, spacing, font... """
    data = {}
    coords = [] if track_coordinates else None
    with open(source_json) as f:
        data = json.load(f)

    code = ""
    for line in data["source-code"]:
        code += line + "\n"

    code = re.sub(r'\n+', '\n', code)

    background_color = (data["background-color"][0],
                        data["background-color"][1],
                        data["background-color"][2])
    width_margin = data["width-margin"]
    height_margin = data["height-margin"]
    spacing = data["spacing"]

    regular = ImageFont.truetype(
        os.path.join(os.getcwd(), font_path, data["font-normal"]), data["font-size"], encoding="unic")
    bold = ImageFont.truetype(
        os.path.join(os.getcwd(), font_path, data["font-bold"]), data["font-size"], encoding="unic")
    italic = ImageFont.truetype(
        os.path.join(os.getcwd(), font_path, data["font-italic"]), data["font-size"], encoding="unic")
    bolditalic = ImageFont.truetype(
        os.path.join(os.getcwd(), font_path, data["font-bold-italic"]), data["font-size"], encoding="unic")

    if logging:
        print(f"code init {code}")
    code_aois = parse_bbcode_list(code, logging)
    actual_code = re.sub(r'\n+', '\n', "".join([l["letter"] for l in code_aois]))
    if logging:
        print(f"code after bbcode for AOIs{actual_code}")

    code_styled = highlight(actual_code, get_lexer_by_name("java"), BBCodeFormatter(
        line_numbers=False, style=get_style_by_name(data["style"] if "style" in data else "vs")))
    code_styled = parse_bbcode_list(code_styled, logging)
    actual_code2 = re.sub(r'\n+', '\n', "".join([l["letter"] for l in code_styled]))
    if logging:
        print(f"code after bbcode for highlights:\n{actual_code2}")
    if len(code_aois) != len(code_styled):
        print('ERROR')
        print(
            f'code does not match:\n from aois"{actual_code}"\n from styling"{actual_code2}"')
        print(
            f'difference in length:\n"{len(actual_code)}"\n"{len(actual_code2)}"')
        # raise Exception()
    code = []
    for idx, c in enumerate(code_styled):
        if logging:
            print('styled', c)
            print('AOI', code_aois[idx])
        code.append({"letter": c["letter"], "color": c["color"], "bold": c["bold"],
                     "italic": c["italic"], "AOI": code_aois[idx]["AOI"]})
    if logging:
        print(f"code after highlight {code}")

    code = set_font(code, regular, bold, italic, bolditalic)
    if logging:
        print(f"code after set_font {code}")

    source_code = []
    tmp_source_code = ""
    for tmp_data in code:
        if tmp_data["letter"] == "\n":
            source_code.append(tmp_source_code)
            tmp_source_code = ""
        else:
            tmp_source_code += tmp_data["letter"]

    img = Image.new("RGBA", (1, 1), color="white")
    draw = ImageDraw.Draw(img)
    max_length = max([
        draw.textbbox((0, 0), txt, font=regular)[2] - draw.textbbox((0, 0), txt, font=regular)[0]
        for txt in source_code
    ]) + 2 * width_margin

    max_height = (
            sum([
                (draw.textbbox((0, 0), "placeholder", font=regular)[3] -
                 draw.textbbox((0, 0), "placeholder", font=regular)[1]) + spacing
                for _ in data["source-code"]
            ]) + 2 * height_margin
    )

    img = Image.new("RGBA", (max_length, max_height), color=background_color)
    draw = ImageDraw.Draw(img)

    width_current = width_margin
    height_current = height_margin
    result = []
    word_start = 0
    in_word = False
    for letter in code:
        if letter["letter"] == "\n":
            if track_coordinates and in_word:
                y_end = int(height_current + (draw.textbbox((0, 0), "placeholder", regular)[3]))
                coords.append([word_start, height_current, width_current, y_end])
                in_word = False
                word_start = 0
            width_current = width_margin
            height_current = height_current + (
                    draw.textbbox((0, 0), "placeholder", font=regular)[3] -
                    draw.textbbox((0, 0), "placeholder", font=regular)[1]
            ) + spacing
            letter["BoundingBox"] = (0, 0, 0, 0)
        else:
            if track_coordinates:
                if letter["letter"].isspace() and in_word:
                    y_end = int(height_current + (draw.textbbox((0, 0), "placeholder", regular)[3]))
                    coords.append([word_start, height_current, width_current, y_end])
                    in_word = False
                    word_start = 0
                elif not letter["letter"].isspace() and not in_word:
                    word_start = width_current
                    in_word = True
            letter["BoundingBox"] = draw.textbbox(
                (width_current,
                 height_current), letter["letter"], font=letter["font"]
            )
            draw.text((width_current, height_current),
                      letter["letter"], letter["color"], font=letter["font"])
            width_current += draw.textbbox((0, 0), letter["letter"], font=regular)[2] - \
                             draw.textbbox((0, 0), letter["letter"], font=regular)[0]

        result.append(letter)
    return img, result, coords


def match_aoi_name_color(aoi_name, mapping):
    """ determine aoi color baased on name"""
    AOI_DEFAUT_COLOR = "#ffffff"
    if aoi_name in mapping:
        return mapping[aoi_name]
    return AOI_DEFAUT_COLOR


def generate_AOI_boxes(image, aoi_list,
                       color_map,
                       padding,
                       remove_nums=False,
                       draw_boxes=True,
                       get_positions=True):
    """adds boxes for aois into image"""
    positions = [] if get_positions else None
    width, height = image.size
    aoi_clustered = []
    current_left = []
    current_top = []
    current_right = []
    current_bottom = []
    current_aoi = []
    color = []
    for letter in aoi_list:
        # Close AOI
        if letter['letter'] in "\t\n":
            continue
        if len(letter["AOI"]) == 1:
            for idx in range(len(current_aoi)):
                aoi_clustered.append(
                    (len(aoi_clustered), current_aoi[idx], current_left[idx], current_top[idx], current_right[idx],
                     current_bottom[idx], color[idx]))

            current_aoi = []
            color = []
            current_left = []
            current_top = []
            current_right = []
            current_bottom = []
            continue
        # There is no AOI set
        if len(current_aoi) == 0:
            current_aoi = []
            color = []
            current_left = []
            current_top = []
            current_right = []
            current_bottom = []
            for idx in range(1, len(letter["AOI"])):
                current_aoi.append(letter["AOI"][idx])
                current_left.append(letter["BoundingBox"][0])
                current_top.append(letter["BoundingBox"][1])
                current_right.append(letter["BoundingBox"][2])
                current_bottom.append(letter["BoundingBox"][3])
                color.append(letter["color"])
            continue

        for idx in reversed(range(len(current_aoi))):
            if current_aoi[idx] in letter["AOI"]:
                current_left[idx] = min(
                    current_left[idx], letter["BoundingBox"][0])
                current_top[idx] = min(
                    current_top[idx], letter["BoundingBox"][1])
                current_right[idx] = max(
                    current_right[idx], letter["BoundingBox"][2])
                current_bottom[idx] = max(
                    current_bottom[idx], letter["BoundingBox"][3])
                # remove the AOI from the letter
                letter["AOI"].remove(current_aoi[idx])
            else:
                aoi_clustered.append(
                    (len(aoi_clustered), current_aoi[idx], current_left[idx], current_top[idx], current_right[idx],
                     current_bottom[idx], color[idx]))
                del current_aoi[idx]
                del current_left[idx]
                del current_top[idx]
                del current_right[idx]
                del current_bottom[idx]
                del color[idx]

        for idx in range(1, len(letter["AOI"])):
            current_aoi.append(letter["AOI"][idx])
            current_left.append(letter["BoundingBox"][0])
            current_top.append(letter["BoundingBox"][1])
            current_right.append(letter["BoundingBox"][2])
            current_bottom.append(letter["BoundingBox"][3])
            color.append(letter["color"])

    for idx, c_a in enumerate(current_aoi):
        aoi_clustered.append((idx, c_a, current_left[idx],
                              current_top[idx], current_right[idx], current_bottom[idx], color[idx]))

    for aoi in aoi_clustered:
        aoi_name = aoi[1]
        while remove_nums and aoi_name:
            if aoi_name[-1].isdigit():
                aoi_name = aoi_name[:-1]
            else:
                break
        outline_color = match_aoi_name_color(aoi_name, color_map)
        image_draw = ImageDraw.Draw(image)
        left_value = max(0, aoi[2] - padding)
        top_value = max(0, aoi[3] - padding)
        right_value = min(width - 1, aoi[4] + padding)
        bottom_value = min(height - 1, aoi[5] + padding)
        if draw_boxes:
            image_draw.rectangle([left_value,
                                  top_value,
                                  right_value,
                                  bottom_value],
                                 outline=outline_color, fill=None, width=4)
        if (get_positions):
            positions.append([left_value, top_value, right_value, bottom_value])
    return positions


def generate_word_dots(image, coords):
    image_draw = ImageDraw.Draw(image)
    for c in coords:
        point_coord = ([(c[0], c[1]), (c[0], c[3]), (c[2], c[1]), (c[2], c[3])])
        image_draw.point(point_coord, fill="#ff0000")


def create_images_from_code(source_folder: str | Path,
                            generator_folder: str | Path,
                            target_folder: str | Path,
                            sub_folders: list[str] = None,
                            get_word_coords: bool = True,
                            get_aoi_positions: bool = True,
                            create_aoi_markers: bool = True,
                            create_coord_markers: bool = True,
                            background_color=None,
                            font_size: int = 30,
                            width_margin: int = 5,
                            height_margin: int = 5,
                            spacing: int = 35,
                            image_size: tuple[int, int] = (1144, 720),
                            highlighting_style: str = 'vs',
                            space_replacement_number: int = 2,
                            font_normal: str = 'JetBrainsMono-Regular.ttf',
                            font_bold: str = 'JetBrainsMono-Bold.ttf',
                            font_italic: str = 'JetBrainsMono-Italic.ttf',
                            font_bold_italic: str = 'JetBrainsMono-BoldItalic.ttf',
                            remove_numbers_from_aois: bool = False,
                            padding: int = 5, aoi_colors: dict[str, str] = None):
    """Function to create Images for Code with AOI from JSON. Parameters: Source_Folder: Relative or absolute path to folder with source code of Snippets.
    * Generator_Folder: Relative or absolute path to folder in which JSONs used to generate images will be stored. Folder will be created if it does not exist.
    * Target_Folder: Relative or absolute path to folder where generated images will be stored. Folder will be created if it does not exist.
    * Sub_Folders: List of subfolders in Source_Folder, should include all subfolders which contain source code which is to be used. Alternatively None if source code is directly in Source_Folder. Default is None.
    * GetWordCoords: Select if edge coordinates of individual words should be given as an output. Default is True
    * GetAOIPositions: Select if edge coordinates of each AOI should be given as an output. Default is True.
    * CreateAOIMarkers: Select if AOI boxes should be drawn in the generated images. Default is True.
    * CreateCoordMarker: Select if red dots to mark Words should be drawn in the generated Images. Default value is True.
    * Lexer: Lexer from pygments.lexers for the programming language in which the snippets in Source_Folder are written.
    * Background_color: List of three integers used as background color of the generated images as [R,G,B]. Default is [255,255,255] (White).
    * Font_size: Size of Font in generated image. Default is 30.
    * Width_margin: Margin between beginning of code and left edge of image. Default is 5.
    * Height_margin: Margin between first line of code and top edge of image. Default is 5.
    * Spacing: Integer indicating the distance between two lines of code. Default is 35.
    * Image_Size: Tuple of Integers indication size of generated Images in Pixels. Default is (1144, 720)
    * Highlighting_Style: String indicating code highlighting style used in image generation. Default is vs. Available styles are those that can be found by get_style_by_name() from pygments.styles.
    * Space_replacement_number: Integer indicating the number of whitespaces with which individual whitespaces will be replaced in the generated images to ensure sufficient space between words. Default is 2.
    * font_normal, font_bold, font_italic,font_bold_italic: Strings indicating which font to use for normal, bold, italic and bold italic parts of code according to code highlighting style. Default are JetBrainsMono fonts.
    * font_path: Path to folder with fonts used for code. Fonts need to be in this folder in .ttf format. Default is ./fonts/ttf/
    * remove_Numbers_from_AOIs: Select if ending numbers in AOI names should be deleted when generating boxes. Can be used to ensure overlapping AOIs work properly. Default is True
    * AOI_Padding: A dictionary for how much padding to add for each AOI Type. Should contain a match for every type used, mandatory parameter
    * AOI_Colors: The Colors to use for marking each AOI, given as a dictionary from AOI names to strings of hexadecimal RGB values. Mandatory Parameter
    Outputs: Two Outputs, the first are AOI coordinates, the second are word coordinates. Both are formatted as lists with one entry per snippet. Each entry is a list with one entry per word/AOI holding coordinates of the edges of the AOI/Word as a list [left, top, right, bottom]"""

    if background_color is None:
        background_color = [255, 255, 255]
    word_coords = {} if get_word_coords else None
    aoi_coords = {} if get_aoi_positions else None
    # Patterns to maipulate whitespace in code
    aoi_tag_pattern = re.compile(r'^([aoi=[^]]+])?( *)')
    single_space_pattern = re.compile('([^ ]) ([^ ])')
    single_space_replacement = r'\1' + ' ' * space_replacement_number + r'\2'
    # Create directories
    os.makedirs(generator_folder, exist_ok=True)
    os.makedirs(target_folder, exist_ok=True)
    if sub_folders is not None:
        for folder in sub_folders:
            os.makedirs(f'{generator_folder}/{folder}', exist_ok=True)
            os.makedirs(f'{target_folder}/{folder}', exist_ok=True)
    data_template = {
        'font-normal': font_normal,
        'font-bold': font_bold,
        'font-italic': font_italic,
        'font-bold-italic': font_bold_italic,
        'font-size': font_size,
        'width-margin': width_margin,  # buffer in width
        'height-margin': height_margin,  # buffer in height
        'spacing': spacing,  # distance between code lines
        'background-color': background_color,  # White
        'style': highlighting_style}
    if sub_folders is None:
        sub_folders = [""]
    for folder in sub_folders:
        # Manipulate and store Source Code from files
        data_dict = {}
        files = os.listdir(f'{source_folder}/{folder}')
        for file in files:
            with open(f'{source_folder}/{folder}/{file}', 'r', encoding='utf-8') as f:
                file_data = f.read()
            file_data_splitted = file_data.split('\n')
            filename, _ = os.path.splitext(file)
            lines = []
            for line in file_data_splitted:
                m = aoi_tag_pattern.match(line)
                if m:
                    aoi_tag = m.group(1) or ''
                    rest = line[m.end():]  # remove all spaces after tag
                    new_line = aoi_tag + rest
                else:
                    new_line = line.lstrip()
                # enlarge spaces (2 instead of 1 per space between tokens)
                old_line, new_line2 = '', new_line
                while new_line2 != old_line:
                    old_line = new_line2
                    new_line2 = re.sub(single_space_pattern, single_space_replacement, old_line)
                lines.append(new_line2)
            data_dict[filename] = lines

        # Write image creation information to generate picture in json files
        for key, value_array in data_dict.items():
            template = data_template.copy()
            template['source-code'] = value_array
            # template['programming-language'] = programming_language
            with open(f'{generator_folder}/{folder}/{key}.json', 'w', encoding='utf-8') as f:
                json.dump(template, f, indent=4, separators=(',', ': '))
        # Take image creation information from json files to generate image
        files = os.listdir(f'{generator_folder}/{folder}')
        img_sizes = {}
        for file in files:
            filename, _file_extension = os.path.splitext(file)
            try:
                img, letters, coords = create_image(f'{generator_folder}/{folder}/{file}',
                                                    track_coordinates=(get_word_coords or create_coord_markers))
                if get_word_coords:
                    word_coords[filename] = coords
            except Exception as e:
                print(f'{file} failed with error: {e}')
                continue
            img_sizes[filename] = img.size
            if any([b['AOI'] != ['None'] for b in letters]) and (create_aoi_markers or get_aoi_positions):
                # print(letters)
                aoi_c = (generate_AOI_boxes(img, letters, aoi_colors, padding, remove_numbers_from_aois,
                                            create_aoi_markers, get_aoi_positions))
                if get_aoi_positions:
                    aoi_coords[filename] = aoi_c
            if create_coord_markers:
                generate_word_dots(img, coords)
            img_sizes[filename] = img.size
            # fixed image size
            img_fixed = Image.new('RGB', image_size, tuple(
                background_color))
            img_fixed.paste(img, (0, 0))
            img = img_fixed
            img.save(f'{target_folder}/{folder}/{filename}.png')
        max_image_size = (max([x_size for x_size, _ in img_sizes.values()]), max(
            [y_size for _, y_size in img_sizes.values()]))
        print(folder, max_image_size)
        if image_size[0] < max_image_size[0]:
            print('image too large, create bigger max values',
                  image_size, max_image_size)
        if image_size[1] < max_image_size[1]:
            print('image too large, create bigger max values',
                  image_size, max_image_size)
    return aoi_coords, word_coords
