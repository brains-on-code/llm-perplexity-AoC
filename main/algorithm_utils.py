import os
import re

import matplotlib.pyplot as plt
import pandas as pd


def detect_perplexity_peaks_scipy(tokens, perplexities, min_span=2, distance=2, prominence=0.5):
    import numpy as np
    from scipy.signal import find_peaks

    y = np.log10(perplexities)
    peaks, properties = find_peaks(y, distance=distance, prominence=prominence)
    peaks = list(peaks)

    # Check last element
    if len(y) > 1 and y[-1] > y[-2] and (y[-1] - y[-2]) >= prominence:
        if len(y) - 1 not in peaks:
            peaks.append(len(y) - 1)

    aois = []
    for peak in peaks:
        left = max(0, peak - min_span // 2)
        right = min(len(tokens) - 1, peak + min_span // 2)
        span_tokens = tokens[left:right + 1]
        if all(t.strip() == '' for t in span_tokens):
            continue
        aois.append({
            "start": left,
            "end": right,
            "tokens": span_tokens,
            "perplexities": perplexities[left:right + 1],
            "peak_index": peak,
            "peak_value": perplexities[peak]
        })

    return aois


def is_identifier(token):
    return re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token) is not None


def is_number(token):
    return re.fullmatch(r"-?\d+(\.\d+)?", token) is not None


def is_operator(token):
    return token in {
        '++', '--', '+', '-', '*', '/', '%',
        '==', '!=', '<', '>', '<=', '>=', '+=', '-=',
        '&&', '||', '!', '&', '|', '^', '~', '<<', '>>'
    }


def combine_tokens_semantically(tokens):
    """
    Combine tokens like ['v', '1', '++'] -> ['v1++'], ['--', 'v', '1'] -> ['--v1'], etc.
    Returns:
        combined_tokens: list of merged token strings
        index_map: list of (start_idx, end_idx) pairs for each merged token
    """
    combined_tokens = []
    index_map = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        j = i
        merged = True
        while merged:
            merged = False
            # Merge identifier + number (e.g., v + 1 -> v1)
            if is_identifier(t) and j + 1 < len(tokens) and is_number(tokens[j + 1]):
                t += tokens[j + 1]
                j += 1
                merged = True
            # Merge decimal numbers (e.g., 1 . 5 -> 1.5)
            elif is_number(t) and j + 2 < len(tokens) and tokens[j + 1] == '.' and is_number(tokens[j + 2]):
                t += tokens[j + 1] + tokens[j + 2]
                j += 2
                merged = True
            # Merge post-increment/decrement (e.g., v1 ++ -> v1++)
            elif (is_identifier(t) or is_number(t)) and j + 1 < len(tokens) and tokens[j + 1] in {'++', '--', '++;',
                                                                                                  '--;'}:
                t += tokens[j + 1]
                j += 1
                merged = True
            # Merge pre-increment/decrement (e.g., -- v1)
            elif t in {'++', '--'} and j + 1 < len(tokens) and (
                    is_identifier(tokens[j + 1]) or is_number(tokens[j + 1])):
                t += tokens[j + 1]
                j += 1
                merged = True

        combined_tokens.append(t)
        index_map.append((i, j))
        i = j + 1

    return combined_tokens, index_map


def expand_aois_semantically(aois, tokens, semantic_tokens, index_map):
    expanded_aois = []

    for aoi in aois:
        aoi_start = aoi["start"]
        aoi_end = aoi["end"]

        # Find semantic token covering aoi_start
        semantic_start = None
        for i, (s, e) in enumerate(index_map):
            if s <= aoi_start <= e:
                semantic_start = i
                break
        if semantic_start is None:
            semantic_start = max(0,
                                 next((i for i, (s, _) in reversed(list(enumerate(index_map))) if s <= aoi_start), 0))

        # Find semantic token covering aoi_end
        semantic_end = None
        for i, (s, e) in reversed(list(enumerate(index_map))):
            if s <= aoi_end <= e:
                semantic_end = i
                break
        if semantic_end is None:
            semantic_end = min(len(index_map) - 1,
                               next((i for i, (_, e) in enumerate(index_map) if e >= aoi_end), len(index_map) - 1))

        # Expand left and right as long as the pattern is [number|identifier] [operator] [number|identifier] ...
        left = semantic_start
        right = semantic_end

        # Expand left
        while left > 0:
            if is_operator(semantic_tokens[left - 1]) and (
                    is_identifier(semantic_tokens[left]) or is_number(semantic_tokens[left])):
                left -= 1
            elif (is_identifier(semantic_tokens[left - 1]) or is_number(semantic_tokens[left - 1])) and is_operator(
                    semantic_tokens[left]):
                left -= 1
            else:
                break

        # Expand right
        while right + 1 < len(semantic_tokens):
            if is_operator(semantic_tokens[right]) and (
                    is_identifier(semantic_tokens[right + 1]) or is_number(semantic_tokens[right + 1])):
                right += 1
            elif (is_identifier(semantic_tokens[right]) or is_number(semantic_tokens[right])) and is_operator(
                    semantic_tokens[right + 1]):
                right += 1
            else:
                break

        orig_start = index_map[left][0]
        orig_end = index_map[right][1]

        expanded_aois.append({
            "start": orig_start,
            "end": orig_end,
            "tokens": tokens[orig_start:orig_end + 1],
            "peak_index": aoi.get("peak_index"),
            "peak_value": aoi.get("peak_value")
        })

    return expanded_aois


def merge_adjacent_aois(aois, tokens):
    """Merge AOIs that are overlapping or directly adjacent."""
    aois = sorted(aois, key=lambda x: x["start"])
    merged = []

    for aoi in aois:
        if not merged:
            merged.append(aoi)
        else:
            last = merged[-1]
            if aoi["start"] <= last["end"] + 1:
                new_start = min(last["start"], aoi["start"])
                new_end = max(last["end"], aoi["end"])
                merged[-1] = {
                    "start": new_start,
                    "end": new_end,
                    "tokens": tokens[new_start:new_end + 1],
                    "peak_index": last.get("peak_index"),
                    "peak_value": max(last.get("peak_value", 0), aoi.get("peak_value", 0))
                }
            else:
                merged.append(aoi)

    return merged


def plot_token_perplexity(results, file, output_path=None, max_token_display=60):
    """
    Compare token-by-token perplexity for a pair of clean and obfuscated files,
    with distinct AOI region highlighting for clean vs obfuscated code.
    """
    # Extract tokens and perplexities from the two files
    name = os.path.basename(file)

    data = pd.DataFrame(results[file]["results"])

    tokens = data['token'].tolist()
    perplexities = data['perplexity'].tolist()
    positions = data['position'].tolist()

    # Get AOI regions from results
    aois = results[file]["aoi_regions"]
    aois_algo = results[file]["aoi_regions_algorithm"]

    # Create a figure for token-by-token comparison
    fig, ax1 = plt.subplots(figsize=(15, 6), sharex=False)

    # Plot perplexities - always plot ALL points
    ax1.semilogy(positions, perplexities,
                 label=name, color='blue', alpha=0.7, marker='o', markersize=4)

    # Define distinct colors and patterns for clean vs obfuscated AOIs
    aoi_color = (0.5, 0.1, 0.1, 0.2)  # Light red
    aoi_algo_color = (0.1, 0.5, 0.5, 0.2)  # Light blue

    # Highlight regions for clean file with brighter colors and dotted patterns
    # clean aois is a dict with aoi type as key and a list of lists with start and end positions
    # e.g. {'BLOCK': [[start1, end1], [start2, end2]]}
    # Iterate through the AOI regions for the file
    for region in aois.values():
        for start, end in region:
            ax1.axvspan(start, end, color=aoi_color, alpha=0.5, label='AOI')

    for region in aois_algo.values():
        for start, end in region:
            ax1.axvspan(start, end, color=aoi_algo_color, alpha=0.5, label='AOI Algorithm')

    # Only show token labels if the number of tokens is less than or equal to max_token_display
    if len(tokens) <= max_token_display:
        # Calculate how many tokens to display
        token_interval = max(1, len(tokens) // max_token_display)
        token_positions = []
        token_labels = []

        for pos in range(0, len(tokens), token_interval):
            token_positions.append(pos)
            if pos < len(tokens):
                # Use a line break to stack tokens vertically
                token = tokens[pos].strip()
                # Truncate long tokens
                if len(token) > 6:
                    token = token[:4] + "..."
                token_labels.append(f"{token}")

        # Set the x-ticks to show token information for axis 1
        ax1.set_xticks(token_positions)
        ax1.set_xticklabels(token_labels)
    else:
        # Don't show any token labels when there are too many tokens
        tick_interval = max(1, len(tokens) // 10)
        ax1.set_xticks(range(0, len(tokens), tick_interval))

    # get max perplexity inside the aoi regions
    max_aoi = max([
        perplexities[start] if start == end else max(perplexities[start:end + 1])
        for region in aois.values()
        for start, end in region
    ])

    # get max perplexity inside the algorithmically detected aoi regions
    max_aoi_algo = [
        perplexities[start] if start == end else max(perplexities[start:end + 1])
        for region in aois_algo.values()
        for start, end in region
    ]

    # Create custom legend entries with max values
    aoi_legend = plt.Line2D([0], [0], color=aoi_color, lw=4,
                            label=f"AOI (max={max_aoi:.0f})")

    # For AOI Algorithm entries, include each region's max
    algo_legend_entries = [
        plt.Line2D([0], [0], color=aoi_algo_color, lw=4,
                   label=f"AOI Algo {i + 1} (max={val:.0f})")
        for i, val in enumerate(max_aoi_algo)
    ]

    # Combine all legend entries
    handles = [aoi_legend] + algo_legend_entries
    ax1.legend(handles=handles, loc='upper right', fontsize=10)

    ax1.set_title(f'Token-by-Token Perplexity: {name}')
    ax1.set_xlabel('Token Position')
    ax1.set_ylabel('Perplexity (log scale)')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-1, len(tokens))

    # Check if algo region contains pre-determined region by checking if any value is contained
    algo_contains_aoi = any(
        any(start_algo <= value <= end_algo
            for value in range(start_aoi, end_aoi + 1)
            for start_algo, end_algo in algo_region)
        for aoi_region in aois.values()
        for start_aoi, end_aoi in aoi_region
        for algo_region in aois_algo.values()
    )

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved token perplexity plot to {output_path}")
    else:
        plt.show()

    return algo_contains_aoi


def insert_aoi_tags_into_code(code, aoi_regions, tokenizer):
    """
    Insert AOI tags like [aoi=1]...[/aoi] into the original source code.
    Token indices from Hugging Face tokenizer are used to locate text spans.
    """
    encoding = tokenizer(code, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoding['offset_mapping']  # List of (start_char, end_char)

    # Flatten into char positions for AOIs
    inserts = []  # list of (char_position, tag_string)

    for idx, (start_token, end_token) in enumerate(aoi_regions, 1):
        start_char = offsets[start_token][0]
        end_char = offsets[end_token][1]

        inserts.append((start_char, f"[aoi={idx}]"))
        inserts.append((end_char, "[/aoi]"))

    # Sort inserts by position (descending so insertion doesn't shift positions)
    inserts.sort(reverse=True, key=lambda x: x[0])

    # Apply the insertions
    code_chars = list(code)
    for pos, tag in inserts:
        code_chars.insert(pos, tag)

    return ''.join(code_chars)


def find_largest_fitting_node(root_node, aoi_start, aoi_end, tolerance=0):
    best_node = None

    def dfs(node):
        nonlocal best_node
        start_byte, end_byte = node.start_byte, node.end_byte

        # Relaxed condition: allow end_byte up to aoi_end + tolerance
        if start_byte >= aoi_start and end_byte <= aoi_end + tolerance:
            if not best_node or (end_byte - start_byte) > (best_node.end_byte - best_node.start_byte):
                best_node = node
        for child in node.children:
            dfs(child)

    dfs(root_node)
    return best_node


def categorize_label(label):
    """
    Comprehensive categorization of Java AST node types from tree-sitter-java
    """

    # Declaration nodes
    if label in {
        "class_declaration", "interface_declaration", "enum_declaration", "annotation_type_declaration",
        "record_declaration", "method_declaration", "constructor_declaration", "field_declaration",
        "constant_declaration", "enum_constant", "annotation_type_element_declaration"
    }:
        return "Declaration"

    # Variable and assignment related
    if label in {
        "variable_declarator", "assignment_expression", "local_variable_declaration",
        "enhanced_for", "resource_specification", "resource"
    }:
        return "Variable / Assignment"

    # Type related
    if label in {
        "integral_type", "boolean_type", "floating_point_type", "void_type",
        "array_type", "generic_type", "type_parameter", "type_arguments", "type_parameters",
        "wildcard", "super_type", "extends_type", "type_bound", "class_type",
        "interface_type", "scoped_type_identifier", "type_identifier"
    }:
        return "Type"

    # Expressions
    if label in {
        "binary_expression", "unary_expression", "update_expression", "parenthesized_expression",
        "conditional_expression", "assignment_expression", "instanceof_expression",
        "lambda_expression", "array_access", "field_access", "method_reference",
        "cast_expression", "array_creation_expression", "object_creation_expression",
        "class_literal", "this", "super"
    }:
        return "Expression"

    # Method/function calls
    if label in {
        "method_invocation", "explicit_constructor_invocation", "super_constructor_invocation"
    }:
        return "Function Call"

    # Control flow
    if label in {
        "if", "else", "while", "for", "enhanced_for",
        "do", "break", "continue", "return",
        "switch", "switch_block", "switch_block_group",
        "switch_label", "case", "default", "yield"
    }:
        return "Control Flow"

    # Exception handling
    if label in {
        "try", "try_with_resources", "catch_clause", "catch_formal_parameter",
        "catch_type", "finally_clause", "throw", "throws"
    }:
        return "Exception Handling"

    # Statements and blocks
    if label in {
        "block", "expression_statement", "empty", "labeled",
        "assert", "synchronized"
    }:
        return "Statement / Block"

    # Identifiers and names
    if label in {
        "identifier", "scoped_identifier", "field_identifier", "type_identifier"
    }:
        return "Identifier"

    # Literals
    if label in {
        "decimal_integer_literal", "hex_integer_literal", "octal_integer_literal",
        "binary_integer_literal", "decimal_floating_point_literal", "hex_floating_point_literal",
        "true", "false", "character_literal", "string_literal", "text_block",
        "null_literal"
    }:
        return "Literal"

    # Operators (terminal symbols)
    if label in {
        "++", "--", "+", "-", "*", "/", "%", "<<", ">>", ">>>",
        "&", "|", "^", "~", "!", "&&", "||", "?", ":",
        "==", "!=", "<", ">", "<=", ">=", "=", "+=", "-=", "*=", "/=", "%=",
        "&=", "|=", "^=", "<<=", ">>=", ">>>=", "->", "::"
    }:
        return "Operator"

    # Modifiers
    if label in {
        "modifiers", "public", "private", "protected", "static", "final", "abstract",
        "synchronized", "native", "strictfp", "transient", "volatile", "default",
        "sealed", "non_sealed"
    }:
        return "Modifier"

    # Annotations
    if label in {
        "annotation", "marker_annotation", "annotation_argument_list",
        "element_value_pair", "element_value_array_initializer"
    }:
        return "Annotation"

    # Import and package
    if label in {
        "import_declaration", "asterisk", "package_declaration"
    }:
        return "Import / Package"

    # Array and collection initializers
    if label in {
        "array_initializer", "variable_declarator_list", "argument_list",
        "formal_parameters", "formal_parameter", "spread_parameter",
        "receiver_parameter"
    }:
        return "Parameter / Argument"

    # Generic/template related
    if label in {
        "type_parameter", "type_parameters", "type_arguments", "wildcard"
    }:
        return "Generic"

    # Comments and documentation
    if label in {
        "line_comment", "block_comment", "javadoc_comment"
    }:
        return "Comment"

    # Program structure
    if label in {
        "program", "compilation_unit"
    }:
        return "Program Structure"

    # Punctuation and delimiters
    if label in {
        "{", "}", "(", ")", "[", "]", ";", ",", ".", "...", "@"
    }:
        return "Punctuation"

    # If none of the above match
    return "Other"
