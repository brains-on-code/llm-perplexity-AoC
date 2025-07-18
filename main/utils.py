import ast
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.ticker import LogLocator, ScalarFormatter
from scipy.stats import pearsonr, spearmanr, kendalltau
from tqdm.notebook import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Metrics to analyze
perplexity_metrics = ['Mean AOI Perplexity', 'Overall AOI Perplexity', 'Max AOI Perplexity']
eeg_metrics = ['mean', 'max']


def read_java_files(base_path):
    """Recursively read all .java files under base_path and return a dictionary of filename"""
    java_files = {}

    for version in os.listdir(base_path):  # e.g., v0, v1, v2
        version_path = os.path.join(base_path, version)
        if os.path.isdir(version_path):  # Ensure it's a directory
            for file in os.listdir(version_path):
                if file.endswith(".java"):
                    file_path = os.path.join(version_path, file)
                    with open(file_path, "r", encoding="utf-8") as f:
                        filename = file_path.split('/')[-1].replace('.java', '')
                        java_files[filename] = f.read()

    return java_files


def parse_aoi_regions(code, aoi_types=None):
    """
    Parse AOI regions from the code and return their positions.

    Returns:
        - cleaned_code: Code with AOI tags removed
        - aoi_regions: Dictionary of AOI regions with their start/end positions
    """
    cleaned_code = []
    aoi_regions = {}
    for aoi_type in aoi_types:
        aoi_regions[aoi_type] = []

    # Stack to handle nested AOIs
    aoi_stack = []

    # Track positions in original and cleaned code
    original_pos = 0
    cleaned_pos = 0

    n = len(code)

    while original_pos < n:
        # Check for opening tag
        if code.startswith('[aoi=', original_pos):
            # Find end of opening tag
            end_tag = code.find(']', original_pos)
            if end_tag == -1:
                # Malformed tag, skip
                cleaned_code.append(code[original_pos])
                cleaned_pos += 1
                original_pos += 1
                continue

            aoi_type = code[original_pos + 5:end_tag]
            if aoi_type in aoi_regions:
                # Push to stack
                aoi_stack.append({
                    'type': aoi_type,
                    'start': cleaned_pos,  # Position in cleaned code
                    'original_start': original_pos
                })
                # Skip the tag in cleaned code
                original_pos = end_tag + 1
            else:
                # Unknown AOI type, treat as normal text
                cleaned_code.append(code[original_pos])
                cleaned_pos += 1
                original_pos += 1

        # Check for closing tag
        elif code.startswith('[/aoi]', original_pos):
            if aoi_stack:
                # Pop from stack
                opened_aoi = aoi_stack.pop()
                aoi_regions[opened_aoi['type']].append({
                    'start': opened_aoi['start'],
                    'end': cleaned_pos,
                    'content': code[opened_aoi['original_start']:original_pos + 6]  # Include tags in content
                })
            # Skip the tag in cleaned code (whether it matched or not)
            original_pos += 6

        else:
            # Normal character
            cleaned_code.append(code[original_pos])
            cleaned_pos += 1
            original_pos += 1

    # Handle any unclosed tags (though they should be properly closed in valid input)
    while aoi_stack:
        opened_aoi = aoi_stack.pop()
        aoi_regions[opened_aoi['type']].append({
            'start': opened_aoi['start'],
            'end': cleaned_pos,
            'content': code[opened_aoi['original_start']:original_pos]
        })

    return ''.join(cleaned_code), aoi_regions


def map_aois_to_perplexity_file(
        input_path, output_path, file_dict_with_aoi_markers, aoi_types
):
    """
    Takes in a JSON file with {file_path: {code, tokens, results}} and augments it
    with AOI token spans and token AOI labels.

    - `file_dict_with_aoi_markers` should contain the original code snippets (with AOI tags)
    - `input_path` is the output from compute_per_token_perplexity
    - `output_path` is where to write the AOI-augmented version
    """
    with open(input_path, "r") as f:
        results = json.load(f)

    for file_name, file_data in tqdm(results.items(), desc="Mapping AOIs"):

        if file_name not in file_dict_with_aoi_markers:
            print(f"Warning: AOI-tagged code not found for {file_name}, skipping.")
            continue

        aoi_code = file_dict_with_aoi_markers[file_name]
        cleaned_code2, aoi_regions = parse_aoi_regions(aoi_code, aoi_types)
        print(f"Cleaned code for {file_name} without AOIs: {cleaned_code2}")
        print(f"Cleaned code for {file_name} from results: {file_data['code']}")
        # Ensure the cleaned code matches the original code in results
        assert cleaned_code2 == file_data["code"], f"Cleaned code mismatch in {file_name}"

        tokens = file_data["tokens"]

        # Compute char offsets for tokens
        char_offsets = []
        current = 0
        for token in tokens:
            char_offsets.append((current, current + len(token)))
            current += len(token)

        # Map AOIs to tokens
        token_aois = {aoi_type: [] for aoi_type in aoi_regions}
        for aoi_type, regions in aoi_regions.items():
            for region in regions:
                start_token = None
                end_token = None
                for i, (start, end) in enumerate(char_offsets):
                    if not (end <= region["start"] or start >= region["end"]):
                        if start_token is None:
                            start_token = i
                        end_token = i
                if start_token is not None and end_token is not None:
                    token_aois[aoi_type].append((start_token, end_token))

        # Annotate each token in the results with AOIs
        for token_result in file_data["results"]:
            pos = token_result["position"]
            token_result["aois"] = []
            for aoi_type, spans in token_aois.items():
                for start, end in spans:
                    if start <= pos <= end:
                        token_result["aois"].append(aoi_type)
                        break

        # Save token AOI regions as well
        file_data["aoi_regions"] = token_aois

    # Save all results with AOIs
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def compute_per_token_perplexity(code, model, tokenizer, device):
    inputs = tokenizer(code, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"].to(device)

    tokens = [tokenizer.decode([t]) for t in input_ids[0]]

    max_length = getattr(model.config, "max_position_embeddings", 1024)
    token_results = []
    log_probs_list = []

    with torch.no_grad():
        bos_token_id = tokenizer.bos_token_id

        # If there's a bos_token, prepend it to the input sequence
        if bos_token_id is not None:
            print(f"Using BOS token: {tokenizer.decode([bos_token_id])}")
            # Prepend bos_token_id to input_ids
            input_ids = torch.cat([torch.tensor([[bos_token_id]], device=device), input_ids], dim=1)
            # Also prepend a decoded token for display, usually something like "<s>"
            tokens = [tokenizer.decode([bos_token_id])] + tokens
        else:
            # Try eos_token_id first if available
            if tokenizer.eos_token_id is not None:
                print(f"Using EOS token: {tokenizer.decode([tokenizer.eos_token_id])}")
                start_token_id = tokenizer.eos_token_id
                start_token_str = tokenizer.decode([start_token_id])
            else:
                # fallback: tokenize a newline as start token
                print("No BOS or EOS token found, using newline as start token")
                newline_ids = tokenizer("\n", add_special_tokens=False).input_ids
                start_token_id = newline_ids[0] if newline_ids else 0
                start_token_str = tokenizer.decode([start_token_id])

            input_ids = torch.cat([torch.tensor([[start_token_id]], device=device), input_ids], dim=1)
            tokens = [start_token_str] + tokens

        seq_len = input_ids.size(1)

        for i in tqdm(range(1, seq_len), desc="Calculating PPL", leave=False):
            start_idx = max(0, i - max_length + 1)
            context_ids = input_ids[:, start_idx:i]
            target_id = input_ids[:, i]

            outputs = model(context_ids)
            logits = outputs.logits[:, -1, :]
            log_probs = torch.log_softmax(logits, dim=-1)

            log_prob = log_probs[0, target_id.item()].item()
            perplexity = math.exp(-log_prob)

            token_results.append({
                "token": tokens[i],
                "position": i - 1,
                "log_prob": log_prob,
                "perplexity": perplexity,
            })

            log_probs_list.append(log_prob)

    overall_perplexity = math.exp(-sum(log_probs_list) / len(log_probs_list))
    mean_perplexity = sum(token["perplexity"] for token in token_results) / len(token_results)
    max_perplexity = max(token["perplexity"] for token in token_results)
    token_count = len(token_results)

    # Add summary statistics
    summary = {
        "overall_perplexity": overall_perplexity,
        "mean_perplexity": mean_perplexity,
        "max_perplexity": max_perplexity,
        "token_count": token_count
    }

    return {
        "code": code,
        "tokens": tokens[1:] if bos_token_id is not None else tokens,  # strip bos token from final output tokens
        "summary": summary,
        "results": token_results
    }


def process_files(file_dict, model_name, save_path="perplexity_results.json", limit=None):
    """
    Process multiple files and compute perplexity for each.

    Args:
        file_dict: Dictionary with file_path as key and code content as value
        model_name: HuggingFace model to use
        save_path: Where to save the results
        limit: Maximum number of files to process (None for all)

    Returns:
        Dictionary with results for each file
    """
    # Load model and tokenizer once
    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading model {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map={"": device}  # Move to MPS
    ).to(device)
    model.eval()

    # Enable padding tokens if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = {}
    count = 0

    # Create progress bar for file processing
    for file_name, code in tqdm(file_dict.items(), desc="Processing files"):
        print(f"Processing: {file_name}")

        # Calculate perplexity
        file_results = compute_per_token_perplexity(
            code, model=model, tokenizer=tokenizer, device=device
        )

        results[file_name] = file_results
        print(f"Completed processing: {file_name}")

        # Save intermediate results
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)

        count += 1
        if limit and count >= limit:
            break

    return results


def find_complementary_pairs(results):
    """
    Find pairs of clean and obfuscated files that correspond to each other

    Args:
        results: Dictionary of file results

    Returns:
        List of tuples (clean_file_path, obf_file_path, snippet_id)
    """
    # Extract file pattern to identify complementary pairs
    clean_files = {}
    obf_files = {}

    for file_path in results.keys():
        file_name = os.path.basename(file_path)

        # Extract the ID from patterns like "13-clean-v1.java" and "13-obf-v1.java"
        parts = file_name.split('-')
        if len(parts) >= 3:
            snippet_id = parts[0]
            version = parts[-1].split('.')[0]  # Extract 'v1' from 'v1.java'
            pair_id = f"{snippet_id}-{version}"

            if "-clean-" in file_name:
                clean_files[pair_id] = file_path
            elif "-obf-" in file_name:
                obf_files[pair_id] = file_path

    # Match the pairs
    pairs = []
    for pair_id, clean_path in clean_files.items():
        if pair_id in obf_files:
            pairs.append((clean_path, obf_files[pair_id], pair_id))

    return pairs


def plot_token_perplexity_for_individual_snippet(results, file, output_path=None, max_token_display=20, t_min=0.4,
                                                 t_max=0.8, eeg_signal=None, eeg_std=None):
    """
    Compare token-by-token perplexity for a pair of clean and obfuscated files,
    with distinct AOI region highlighting for clean vs obfuscated code.
    """
    # Extract tokens and perplexities from the two files
    file_name = os.path.basename(file)

    data = pd.DataFrame(results[file]["results"])

    tokens = data['token'].tolist()
    perplexities = data['perplexity'].tolist()
    log_probs = data['log_prob'].tolist()
    positions = data['position'].tolist()

    # Get AOI regions from results (it should only be one region per file)
    aois = results[file]["aoi_regions"]
    aoi_labels = results[file]["aoi_labels"]
    aoi_label = aoi_labels

    # Create a figure for token-by-token comparison
    if eeg_signal is None:
        fig, ax1 = plt.subplots(figsize=(15, 6), sharex=False)
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=False)
        fig.subplots_adjust(hspace=0.4)  # Adjust space between subplots
    # Plot perplexities - always plot ALL points
    ax1.semilogy(positions, perplexities,
                 label=file_name, color='blue', alpha=0.7, marker='o', markersize=4)

    # Define distinct colors and patterns for clean vs obfuscated AOIs
    aoi_color = (0.1, 0.5, 0.5, 0.2)  # Light blue

    # Highlight regions for clean file with brighter colors and dotted patterns
    # clean aois is a dict with aoi type as key and a list of lists with start and end positions
    # e.g. {'BLOCK': [[start1, end1], [start2, end2]]}
    # Iterate through the AOI regions for the clean file not knowing the key
    # We assume there is only one AOI region per file
    for region in aois.values():
        for start, end in region:
            ax1.axvspan(start, end, color=aoi_color, alpha=0.5, label='AOI')

    # Add a legend for AOI regions
    # Create a custom legend for AOI regions
    patch = plt.Line2D([0], [0], color=aoi_color, lw=4, label='Clean AOI')
    # Add the patches to the legend and perplexity lines
    ax1.legend(handles=[patch], loc='upper left', fontsize=10)
    # Add the perplexity lines to the legend
    ax1.legend(loc='upper right', fontsize=10)
    # Original token display functionality
    max_length = len(tokens)

    # Only show token labels if the number of tokens is less than or equal to max_token_display
    if max_length <= max_token_display:
        # Calculate how many tokens to display
        token_interval = max(1, max_length // 20)  # Show around 20 token positions
        token_positions = []
        token_labels = []

        for pos in range(0, max_length, token_interval):
            token_positions.append(pos)

            if pos < len(tokens):
                clean_token = tokens[pos].strip()
                # Truncate long tokens
                if len(clean_token) > 10:
                    clean_token = clean_token[:8] + "..."
                token_labels.append(f"{clean_token}")

        # Set the x-ticks to show token information for axis 1
        ax1.set_xticks(token_positions)
        ax1.set_xticklabels(token_labels)
    else:
        # Don't show any token labels when there are too many tokens
        tick_interval = max(1, max_length // 10)
        ax1.set_xticks(range(0, max_length, tick_interval))

    # Original statistics functionality
    stats = results[file]["summary"]
    # get max perplexity inside the aoi regions
    max_aoi = max([
        perplexities[start] if start == end else max(perplexities[start:end + 1])
        for region in aois.values()
        for start, end in region
    ])

    # get mean perplexity inside the aoi regions
    mean_aoi = np.mean([
        perplexities[start] if start == end else
        np.mean(perplexities[start:end + 1]) for region in aois.values() for start, end in region
    ])

    # get the overall perplexity inside the aoi regions
    aoi_log_probs = [
        log_probs[start] if start == end else
        np.mean([log_probs[i] for i in range(start, end + 1)])
        for region in aois.values() for start, end in region
    ]

    # Compute overall AOI perplexity
    overall_aoi = np.exp(-np.mean(aoi_log_probs))

    # Create a more detailed stats box
    stats_text = (
        f"Overall={stats['overall_perplexity']:.2f}, "
        f"Mean={stats['mean_perplexity']:.2f}, "
        f"Max={stats['max_perplexity']:.2f}, "
        f"Overall(AOI)={overall_aoi:.2f}, "
        f"Mean(AOI)={mean_aoi:.2f}, "
        f"Max(AOI)={max_aoi:.2f}\n"
    )

    # Add the stats text box below the plot with padding
    ax1.text(0.5, -0.25, stats_text, transform=ax1.transAxes, ha='center', fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))
    ax1.set_title(f'Per-Token Perplexity: {file_name}')
    ax1.set_xlabel('Token Position')
    ax1.set_ylabel('Perplexity (log scale)')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-1, max_length)

    if output_path:
        to_remove = output_path.split('/')[-1]
        # Remove the file name from the path
        base_path = output_path.replace(to_remove, "")
        if os.path.exists(base_path + "perplexity_stats.csv"):
            stats_df = pd.read_csv(base_path + "perplexity_stats.csv")
        else:
            # Create a new CSV
            stats_df = pd.DataFrame(
                columns=['File', 'Overall Perplexity', 'Mean Perplexity', 'Max Perplexity', 'Overall AOI Perplexity',
                         'Mean AOI Perplexity', 'Max AOI Perplexity', 'AOI Label'])
        # Check if the file already exists
        if file_name in stats_df['File'].values:
            # Update the existing row
            stats_df.loc[stats_df['File'] == file_name, ['Overall Perplexity', 'Mean Perplexity', 'Max Perplexity',
                                                         'Overall AOI Perplexity', 'Mean AOI Perplexity',
                                                         'Max AOI Perplexity', 'AOI Label']] = [
                stats['overall_perplexity'], stats['mean_perplexity'], stats['max_perplexity'],
                overall_aoi, mean_aoi, max_aoi, aoi_label
            ]
        else:
            # Append new rows
            stats_df = pd.concat([
                stats_df,
                pd.DataFrame([{
                    'File': file_name,
                    'Overall Perplexity': stats['overall_perplexity'],
                    'Mean Perplexity': stats['mean_perplexity'],
                    'Max Perplexity': stats['max_perplexity'],
                    'Overall AOI Perplexity': overall_aoi,
                    'Mean AOI Perplexity': mean_aoi,
                    'Max AOI Perplexity': max_aoi,
                    'AOI Label': aoi_labels
                }]),
            ], ignore_index=True)

        # Save the updated DataFrame to CSV
        stats_df.to_csv(base_path + "perplexity_stats.csv", index=False)

    # -------------------- EEG Signal Comparison Plot (Bottom Subplot) -------------------- #
    if eeg_signal is not None and eeg_std is not None:
        # Get the time vector for the EEG signal
        time_vector = np.linspace(t_min, t_max, len(eeg_signal))
        # Plot the clean EEG signal
        ax2.plot(time_vector, eeg_signal, label='Clean EEG Signal', color='blue')

        # Plot the standard deviation as shaded area
        ax2.fill_between(time_vector, eeg_signal - eeg_std, eeg_signal + eeg_std,
                         color='blue', alpha=0.2, label='Clean EEG Std Dev')

        # plot average EEG signal as a horizontal line
        ax2.axhline(y=np.mean(eeg_signal), color='blue', linestyle='--', label='EEG Mean', alpha=0.8)
        # Add a vertical line at the start of the AOI region
        # calculate mean potential for both clean and obfuscated
        mean_potential = np.mean(eeg_signal)
        # Add text box with mean potential as e-6 as it is too small
        mean_text = (
            f"Mean Potential: {mean_potential:.2e}\n"
        )

        # Add the mean potential text box below the plot with padding
        ax2.text(0.5, -0.25, mean_text, transform=ax2.transAxes, ha='center', fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))
        # add legend excluding the std dev areas
        handles, labels = ax2.get_legend_handles_labels()
        # Filter out the std dev areas
        filtered_handles = [h for h, l in zip(handles, labels) if 'Std Dev' not in l]
        filtered_labels = [l for l in labels if 'Std Dev' not in l]
        ax2.legend(filtered_handles, filtered_labels, loc='upper right')
        ax2.set_ylabel('EEG Signal')
        ax2.set_title('EEG Signal')
        ax2.set_xlabel('Time (s)')
        ax2.grid(True, alpha=0.3)

        # Add the mean potential text box below the plot

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig


def compare_token_perplexity_for_pair(results, clean_file, obf_file, output_path=None, max_token_display=20, t_min=0.4,
                                      t_max=0.8, clean_eeg_signal=None, clean_eeg_std=None, obf_eeg_signal=None,
                                      obf_eeg_std=None):
    """
    Compare token-by-token perplexity for a pair of clean and obfuscated files,
    with distinct AOI region highlighting for clean vs obfuscated code.
    """
    # Extract tokens and perplexities from the two files
    clean_data = pd.DataFrame(results[clean_file]["results"])
    obf_data = pd.DataFrame(results[obf_file]["results"])

    clean_tokens = clean_data['token'].tolist()
    clean_perplexities = clean_data['perplexity'].tolist()
    clean_log_probs = clean_data['log_prob'].tolist()
    clean_positions = clean_data['position'].tolist()

    obf_tokens = obf_data['token'].tolist()
    obf_perplexities = obf_data['perplexity'].tolist()
    obf_log_probs = obf_data['log_prob'].tolist()
    obf_positions = obf_data['position'].tolist()

    # Get AOI regions from results (it should only be one region per file)
    clean_aois = results[clean_file]["aoi_regions"]
    obf_aois = results[obf_file]["aoi_regions"]

    # Create a figure for token-by-token comparison
    if clean_eeg_signal is None and obf_eeg_signal is None:
        fig, ax1 = plt.subplots(figsize=(15, 6), sharex=False)
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=False)
        fig.subplots_adjust(hspace=0.4)  # Adjust space between subplots
    # Plot perplexities - always plot ALL points
    ax1.semilogy(clean_positions, clean_perplexities,
                 label=clean_file, color='blue', alpha=0.7, marker='o', markersize=4)
    ax1.semilogy(obf_positions, obf_perplexities,
                 label=obf_file, color='red', alpha=0.7, marker='x', markersize=4)

    # Define distinct colors and patterns for clean vs obfuscated AOIs
    clean_aoi_color = (0.1, 0.5, 0.5, 0.2)  # Light blue 

    obf_aoi_color = (0.5, 0.1, 0.1, 0.2)  # Light red

    # Highlight regions for clean file with brighter colors and dotted patterns
    # clean aois is a dict with aoi type as key and a list of lists with start and end positions
    # e.g. {'BLOCK': [[start1, end1], [start2, end2]]}
    # Iterate through the AOI regions for the clean file not knowing the key
    # We assume there is only one AOI region per file
    for region in clean_aois.values():
        for start, end in region:
            ax1.axvspan(start, end, color=clean_aoi_color, alpha=0.5, label='Clean AOI')

    # Highlight regions for obfuscated file with brighter colors and dotted patterns
    for region in obf_aois.values():
        for start, end in region:
            ax1.axvspan(start, end, color=obf_aoi_color, alpha=0.5, label='Obfuscated AOI')

    # Add a legend for AOI regions
    # Create a custom legend for AOI regions
    clean_patch = plt.Line2D([0], [0], color=clean_aoi_color, lw=4, label='Clean AOI')
    obf_patch = plt.Line2D([0], [0], color=obf_aoi_color, lw=4, label='Obfuscated AOI')
    # Add the patches to the legend and perplexity lines
    ax1.legend(handles=[clean_patch, obf_patch], loc='upper left', fontsize=10)
    # Add the perplexity lines to the legend
    ax1.legend(loc='upper right', fontsize=10)
    # Original token display functionality
    max_length = max(len(clean_tokens), len(obf_tokens))

    # Only show token labels if the number of tokens is less than or equal to max_token_display
    if max_length <= max_token_display:
        # Calculate how many tokens to display
        token_interval = max(1, max_length // 20)  # Show around 20 token positions
        token_positions = []
        token_labels = []

        for pos in range(0, max_length, token_interval):
            token_positions.append(pos)

            if pos < len(clean_tokens) and pos < len(obf_tokens):
                # Use a line break to stack tokens vertically
                clean_token = clean_tokens[pos].strip()
                obf_token = obf_tokens[pos].strip()

                # Truncate long tokens
                if len(clean_token) > 10:
                    clean_token = clean_token[:8] + "..."
                if len(obf_token) > 10:
                    obf_token = obf_token[:8] + "..."

                token_labels.append(f"{clean_token}\n{obf_token}")
            elif pos < len(clean_tokens):
                token = clean_tokens[pos].strip()
                if len(token) > 10:
                    token = token[:8] + "..."
                token_labels.append(f"{token}\n")
            else:
                token = obf_tokens[pos].strip()
                if len(token) > 10:
                    token = token[:8] + "..."
                token_labels.append(f"\n{token}")

        # Set the x-ticks to show token information for axis 1
        ax1.set_xticks(token_positions)
        ax1.set_xticklabels(token_labels)
    else:
        # Don't show any token labels when there are too many tokens
        tick_interval = max(1, max_length // 10)
        ax1.set_xticks(range(0, max_length, tick_interval))

    # Original statistics functionality
    clean_stats = results[clean_file]["summary"]
    obf_stats = results[obf_file]["summary"]
    # get max perplexity inside the aoi regions
    clean_max_aoi = max([
        clean_perplexities[start] if start == end else max(clean_perplexities[start:end + 1])
        for region in clean_aois.values()
        for start, end in region
    ])
    obf_max_aoi = max([
        obf_perplexities[start] if start == end else max(obf_perplexities[start:end + 1])
        for region in obf_aois.values()
        for start, end in region
    ])

    # get mean perplexity inside the aoi regions
    clean_mean_aoi = np.mean([
        clean_perplexities[start] if start == end else
        np.mean(clean_perplexities[start:end + 1]) for region in clean_aois.values() for start, end in region
    ])
    obf_mean_aoi = np.mean([
        obf_perplexities[start] if start == end else
        np.mean(obf_perplexities[start:end + 1]) for region in obf_aois.values() for start, end in region
    ])

    # get the overall perplexity inside the aoi regions
    clean_aoi_log_probs = [
        clean_log_probs[start] if start == end else
        np.mean([clean_log_probs[i] for i in range(start, end + 1)])
        for region in clean_aois.values() for start, end in region
    ]
    obf_aoi_log_probs = [
        obf_log_probs[start] if start == end else
        np.mean([obf_log_probs[i] for i in range(start, end + 1)])
        for region in obf_aois.values() for start, end in region
    ]

    clean_overall_aoi = np.exp(-np.mean(clean_aoi_log_probs))
    obf_overall_aoi = np.exp(-np.mean(obf_aoi_log_probs))

    # Create a more detailed stats box
    stats_text = (
        f"Clean: Overall={clean_stats['overall_perplexity']:.2f}, "
        f"Mean={clean_stats['mean_perplexity']:.2f}, "
        f"Max={clean_stats['max_perplexity']:.2f}, "
        f"Overall(AOI)={clean_overall_aoi:.2f}, "
        f"Mean(AOI)={clean_mean_aoi:.2f}, "
        f"Max(AOI)={clean_max_aoi:.2f}\n"
        f"Obf: Overall={obf_stats['overall_perplexity']:.2f}, "
        f"Mean={obf_stats['mean_perplexity']:.2f}, "
        f"Max={obf_stats['max_perplexity']:.2f}, "
        f"Overall(AOI)={obf_overall_aoi:.2f}, "
        f"Mean(AOI)={obf_mean_aoi:.2f}, "
        f"Max(AOI)={obf_max_aoi:.2f}"
    )

    # Add the stats text box below the plot with padding
    ax1.text(0.5, -0.25, stats_text, transform=ax1.transAxes, ha='center', fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))
    ratio = obf_overall_aoi / clean_overall_aoi if clean_overall_aoi != 0 else float('inf')
    ax1.set_title(f'Token-by-Token Perplexity: {clean_file} vs {obf_file} (Obf/Clean Ratio (AOI): {ratio:.2f}x)')
    ax1.set_xlabel('Token Position')
    ax1.set_ylabel('Perplexity (log scale)')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-1, max_length)

    if output_path:
        to_remove = output_path.split('/')[-1]
        # Remove the file name from the path
        base_path = output_path.replace(to_remove, "")
        if os.path.exists(base_path + "perplexity_stats.csv"):
            stats_df = pd.read_csv(base_path + "perplexity_stats.csv")
        else:
            # Create a new CSV
            stats_df = pd.DataFrame(
                columns=['File', 'Overall Perplexity', 'Mean Perplexity', 'Max Perplexity', 'Overall AOI Perplexity',
                         'Mean AOI Perplexity', 'Max AOI Perplexity'])
        # Check if the file already exists
        if clean_file in stats_df['File'].values:
            # Update the existing row
            stats_df.loc[stats_df['File'] == clean_file, ['Overall Perplexity', 'Mean Perplexity', 'Max Perplexity',
                                                          'Overall AOI Perplexity', 'Mean AOI Perplexity',
                                                          'Max AOI Perplexity']] = [
                clean_stats['overall_perplexity'], clean_stats['mean_perplexity'], clean_stats['max_perplexity'],
                clean_overall_aoi, clean_mean_aoi, clean_max_aoi
            ]
            stats_df.loc[stats_df['File'] == obf_file, ['Overall Perplexity', 'Mean Perplexity', 'Max Perplexity',
                                                        'Overall AOI Perplexity', 'Mean AOI Perplexity',
                                                        'Max AOI Perplexity']] = [
                obf_stats['overall_perplexity'], obf_stats['mean_perplexity'], obf_stats['max_perplexity'],
                obf_overall_aoi, obf_mean_aoi, obf_max_aoi
            ]
        else:
            # Append new rows
            stats_df = pd.concat([
                stats_df,
                pd.DataFrame([{
                    'File': clean_file,
                    'Overall Perplexity': clean_stats['overall_perplexity'],
                    'Mean Perplexity': clean_stats['mean_perplexity'],
                    'Max Perplexity': clean_stats['max_perplexity'],
                    'Overall AOI Perplexity': clean_overall_aoi,
                    'Mean AOI Perplexity': clean_mean_aoi,
                    'Max AOI Perplexity': clean_max_aoi
                }]),
                pd.DataFrame([{
                    'File': obf_file,
                    'Overall Perplexity': obf_stats['overall_perplexity'],
                    'Mean Perplexity': obf_stats['mean_perplexity'],
                    'Max Perplexity': obf_stats['max_perplexity'],
                    'Overall AOI Perplexity': obf_overall_aoi,
                    'Mean AOI Perplexity': obf_mean_aoi,
                    'Max AOI Perplexity': obf_max_aoi
                }])
            ], ignore_index=True)

        # Save the updated DataFrame to CSV
        stats_df.to_csv(base_path + "perplexity_stats.csv", index=False)

    # -------------------- EEG Signal Comparison Plot (Bottom Subplot) -------------------- #    
    if clean_eeg_signal is not None and obf_eeg_signal is not None and clean_eeg_std is not None and obf_eeg_std is not None:
        # Get the time vector for the EEG signal
        time_vector = np.linspace(t_min, t_max, len(clean_eeg_signal))
        # Plot the clean EEG signal
        ax2.plot(time_vector, clean_eeg_signal, label='Clean EEG Signal', color='blue')
        # Plot the obfuscated EEG signal
        ax2.plot(time_vector, obf_eeg_signal, label='Obfuscated EEG Signal', color='red')

        # Plot the standard deviation as shaded area
        ax2.fill_between(time_vector, clean_eeg_signal - clean_eeg_std, clean_eeg_signal + clean_eeg_std,
                         color='blue', alpha=0.2, label='Clean EEG Std Dev')
        ax2.fill_between(time_vector, obf_eeg_signal - obf_eeg_std, obf_eeg_signal + obf_eeg_std,
                         color='red', alpha=0.2, label='Obfuscated EEG Std Dev')

        # plot average EEG signal as a horizontal line
        ax2.axhline(y=np.mean(clean_eeg_signal), color='blue', linestyle='--', label='Clean EEG Mean', alpha=0.8)
        ax2.axhline(y=np.mean(obf_eeg_signal), color='red', linestyle='--', label='Obfuscated EEG Mean', alpha=0.8)
        # Add a vertical line at the start of the AOI region
        # calculate mean potential for both clean and obfuscated
        clean_mean_potential = np.mean(clean_eeg_signal)
        obf_mean_potential = np.mean(obf_eeg_signal)
        # Add text box with mean potential as e-6 as it is too small
        mean_text = (
            f"Clean Mean Potential: {clean_mean_potential:.2e}\n"
            f"Obfuscated Mean Potential: {obf_mean_potential:.2e}"
        )

        # Add the mean potential text box below the plot with padding
        ax2.text(0.5, -0.25, mean_text, transform=ax2.transAxes, ha='center', fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))
        # add legend excluding the std dev areas
        handles, labels = ax2.get_legend_handles_labels()
        # Filter out the std dev areas
        filtered_handles = [h for h, l in zip(handles, labels) if 'Std Dev' not in l]
        filtered_labels = [l for l in labels if 'Std Dev' not in l]
        ax2.legend(filtered_handles, filtered_labels, loc='upper right')
        ax2.set_ylabel('EEG Signal')
        ax2.set_title('EEG Signal Comparison')
        ax2.set_xlabel('Time (s)')
        ax2.grid(True, alpha=0.3)

        # Add the mean potential text box below the plot

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig


def analyze_all_pairs(results, java_files_dir_with_aois, output_dir=None, averaged_trials=None,
                      std_trials=None, t_min=0.4, t_max=0.8):
    """
    Analyze all complementary pairs of clean/obfuscated code

    Args:
        results: Dictionary of file results
        output_dir: Optional directory to save figures

    Returns:
        Dictionary with pair analysis results
    """
    pairs = find_complementary_pairs(results)
    pair_results = {}

    # read out optimal aoi region for each file
    df = pd.read_csv(os.path.join(java_files_dir_with_aois, "aoi_sizes.csv"), delimiter=";")
    # Filter rows where OPTIMAL is True
    optimal_aoi_df = df[df['OPTIMAL']]
    # look which aoi region is optimal for each file
    optimal_aoi = {}
    for index, row in optimal_aoi_df.iterrows():
        file_name = row['Snippet']
        aoi_type = row['NO']
        optimal_aoi[file_name] = aoi_type

    # keep only the optimal aoi region for each file in the results
    for file_path, file_results in results.items():
        file_name = os.path.basename(file_path).split('.')[0]  # Extract the snippet name
        if file_name in optimal_aoi:
            aoi_type = optimal_aoi[file_name]
            results[file_path]["aoi_regions"] = {aoi_type: results[file_path]["aoi_regions"][aoi_type]}
        else:
            results[file_path]["aoi_regions"] = {}

    for clean_file, obf_file, pair_id in pairs:
        print(f"Analyzing pair: {pair_id}")

        # Generate output path if directory is provided
        output_path = None
        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_path = os.path.join(output_dir, f"pair_{pair_id}_comparison.pdf")

        if averaged_trials is not None and std_trials is not None:
            # Get the EEG signals for the clean and obfuscated files
            clean_eeg_signal = averaged_trials[clean_file][0]
            clean_eeg_std = std_trials[clean_file][0]
            obf_eeg_signal = averaged_trials[obf_file][0]
            obf_eeg_std = std_trials[obf_file][0]
            fig = compare_token_perplexity_for_pair(results, clean_file, obf_file, output_path, t_min=t_min,
                                                    t_max=t_max, clean_eeg_signal=clean_eeg_signal,
                                                    clean_eeg_std=clean_eeg_std, obf_eeg_signal=obf_eeg_signal,
                                                    obf_eeg_std=obf_eeg_std)
        else:
            # Compare token perplexity for this pair
            fig = compare_token_perplexity_for_pair(results, clean_file, obf_file, output_path)

        # Store the pair comparison results
        clean_stats = results[clean_file]["summary"]
        obf_stats = results[obf_file]["summary"]

        pair_results[pair_id] = {
            "clean_file": clean_file,
            "obf_file": obf_file,
            "clean_stats": clean_stats,
            "obf_stats": obf_stats,
            "perplexity_diff": obf_stats["overall_perplexity"] - clean_stats["overall_perplexity"],
            "perplexity_ratio": obf_stats["overall_perplexity"] / clean_stats["overall_perplexity"]
        }

        plt.close(fig)

    # Create summary of all pairs
    if pairs:
        summary_df = pd.DataFrame([{
            'pair_id': pair_id,
            'clean_file': os.path.basename(pair_results[pair_id]['clean_file']),
            'obf_file': os.path.basename(pair_results[pair_id]['obf_file']),
            'clean_perplexity': pair_results[pair_id]['clean_stats']['overall_perplexity'],
            'obf_perplexity': pair_results[pair_id]['obf_stats']['overall_perplexity'],
            'diff': pair_results[pair_id]['perplexity_diff'],
            'ratio': pair_results[pair_id]['perplexity_ratio']
        } for pair_id in pair_results.keys()])

        # Sort by the ratio (obfuscated/clean)
        summary_df = summary_df.sort_values('ratio', ascending=False)

        # Create summary plot
        plt.figure(figsize=(6, 15))
        plt.barh(summary_df['pair_id'], summary_df['ratio'], color='purple')
        plt.title('Obfuscation Impact: Perplexity Ratio (Obfuscated/Clean)')
        plt.xlabel('Ratio')
        plt.ylabel('Code Snippet Pair')
        plt.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()
        if output_dir:
            plt.savefig(os.path.join(output_dir, "all_pairs_ratio_comparison.pdf"), format='pdf', dpi=300)
        plt.close()

        pair_results['summary'] = summary_df


def create_eeg_perplexity_correlation_analysis_df(results_path, output_dir=None, averaged_trials=None,
                                                  max_trials=None):
    """
    Perplexity metrics analyzed:
    - Mean AOI Perplexity
    - Overall AOI Perplexity
    - Max AOI Perplexity
    
    EEG metrics analyzed:
    - Mean EEG signal
    - Max EEG signal
    
    Args:
        results_path: Path to the perplexity results CSV
        output_dir: Directory to save plots
    """
    # Check if the output directory exists, if not create it
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Load perplexity data
    perplexity_df = pd.read_csv(results_path)

    # Extract clean and obfuscated snippets
    clean_snippets = perplexity_df[perplexity_df['File'].str.contains('-clean-')]
    obf_snippets = perplexity_df[perplexity_df['File'].str.contains('-obf-')]

    # Create dataframes for analysis
    analysis_data = []

    # Process each pair of clean and obfuscated snippets
    for clean_file in clean_snippets['File']:
        # Extract snippet ID (e.g., "13-v1" from "13-clean-v1.java")
        parts = clean_file.split('-')
        snippet_id = parts[0]
        version = parts[-1].split('.')[0]
        pair_id = f"{snippet_id}-{version}"

        # Find matching obfuscated file
        obf_file = f"{snippet_id}-obf-{version}"

        if obf_file in obf_snippets['File'].values:
            # Get clean snippet data
            clean_snippet_name = clean_file.split('.')[0]
            if clean_snippet_name in averaged_trials and clean_snippet_name in max_trials:
                # Get mean EEG signal
                clean_mean_eeg = np.mean(averaged_trials[clean_snippet_name][0])
                # Get max EEG signal
                clean_max_eeg = np.mean(max_trials[clean_snippet_name][0])

                # Get metrics for clean snippet
                clean_perplexity_metrics = {}
                for metric in perplexity_metrics:
                    clean_perplexity_metrics[metric] = clean_snippets.loc[clean_snippets['File']
                                                                          == clean_file, metric].values[0]

                # Get obfuscated snippet data
                obf_snippet_name = obf_file.split('.')[0]
                if obf_snippet_name in averaged_trials and obf_snippet_name in max_trials:
                    # Get mean EEG signal
                    obf_mean_eeg = np.mean(averaged_trials[obf_snippet_name][0])
                    # Get max EEG signal
                    obf_max_eeg = np.mean(max_trials[obf_snippet_name][0])

                    # Get metrics for obfuscated snippet
                    obf_perplexity_metrics = {}
                    for metric in perplexity_metrics:
                        obf_perplexity_metrics[metric] = obf_snippets.loc[obf_snippets['File']
                                                                          == obf_file, metric].values[0]

                    # Base entry for this pair
                    entry = {
                        'pair_id': pair_id,
                        'clean_file': clean_file,
                        'obf_file': obf_file,
                        'clean_mean_eeg': clean_mean_eeg,
                        'obf_mean_eeg': obf_mean_eeg,
                        'mean_eeg_diff': obf_mean_eeg - clean_mean_eeg,
                        'clean_max_eeg': clean_max_eeg,
                        'obf_max_eeg': obf_max_eeg,
                        'max_eeg_diff': obf_max_eeg - clean_max_eeg,
                    }

                    # Add perplexity metrics and their differences
                    for metric in perplexity_metrics:
                        entry[f'clean_{metric}'] = clean_perplexity_metrics[metric]
                        entry[f'obf_{metric}'] = obf_perplexity_metrics[metric]
                        entry[f'{metric}_diff'] = obf_perplexity_metrics[metric] - clean_perplexity_metrics[metric]
                        entry[f'{metric}_ratio'] = obf_perplexity_metrics[metric] / clean_perplexity_metrics[metric]
                        entry[f'{metric}_log_ratio'] = math.log(obf_perplexity_metrics[metric]) / math.log(
                            clean_perplexity_metrics[metric])

                    # Add to analysis data
                    analysis_data.append(entry)

    # Convert to DataFrame
    analysis_df = pd.DataFrame(analysis_data)
    # save the analysis DataFrame to a CSV file
    if output_dir:
        output_path = os.path.join(output_dir, "eeg_perplexity_analysis.csv")
        # delete the file if it exists
        if os.path.exists(output_path):
            os.remove(output_path)
        analysis_df.to_csv(output_path, index=False)
        print(f"Saved EEG-perplexity analysis to: {output_path}")

    return analysis_df


def create_eeg_perplexity_correlation_plots(analysis_df, output_dir_scatter_plots=None):
    # Check if the output directory exists, if not create it
    if output_dir_scatter_plots and not os.path.exists(output_dir_scatter_plots):
        os.makedirs(output_dir_scatter_plots)
    correlation_results = []

    def plot_and_corr(x_col, y_col, label, color, fname_prefix):
        x = analysis_df[x_col]
        y = analysis_df[y_col]

        # Only use positive x values for log scale
        valid_x = x[x > 0]
        if valid_x.empty:
            return

        y = y.loc[valid_x.index]

        plt.figure(figsize=(10, 8))
        plt.scatter(valid_x, y, color=color, alpha=0.7, s=100)
        for i, row in analysis_df.loc[valid_x.index].iterrows():
            plt.annotate(row['pair_id'], (row[x_col], row[y_col]), xytext=(5, 5), textcoords='offset points')

        # Log10 scale and ticks
        plt.xscale('log', base=10)
        min_exp = int(np.floor(np.log10(valid_x.min())))
        max_exp = int(np.ceil(np.log10(valid_x.max())))
        xticks = [10 ** i for i in range(min_exp, max_exp + 1)]
        xtick_labels = [r"$10^{%d}$" % i for i in range(min_exp, max_exp + 1)]
        plt.xticks(xticks, xtick_labels)
        plt.gca().xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
        plt.gca().xaxis.set_major_formatter(ScalarFormatter())

        # Trend line in log10 space
        x_log = np.log10(valid_x)
        if len(x_log) > 1:
            coeffs = np.polyfit(x_log, y, 1)
            poly_fn = np.poly1d(coeffs)
            x_range = np.linspace(x_log.min(), x_log.max(), 100)
            plt.plot(10 ** x_range, poly_fn(x_range), "r--", alpha=0.7, label='Trend Line')
            pearson_r, pearson_p = pearsonr(x_log, y)
            spearman_r, spearman_p = spearmanr(valid_x, y)
            kendall_r, kendall_p = kendalltau(valid_x, y)
        else:
            pearson_r = pearson_p = spearman_r = spearman_p = kendall_r = kendall_p = np.nan

        plt.title(
            f'{label} Snippets: {eeg_metric.title()} EEG vs {x_col.replace(f"{label.lower()}_", "")}\n'
            f'Pearson r={pearson_r:.3f} (p={pearson_p:.3f}), '
            f'Spearman r={spearman_r:.3f} (p={spearman_p:.3f}), '
            f'Kendall r={kendall_r:.3f} (p={kendall_p:.3f})')
        plt.xlabel(f'{x_col} (Log$_{{10}}$ Scale)')
        plt.ylabel(f'{y_col}')
        plt.grid(True, alpha=0.3)
        plt.legend()

        if output_dir_scatter_plots:
            plt.savefig(f"{output_dir_scatter_plots}/{fname_prefix}_{y_col}_vs_{x_col}.pdf", bbox_inches='tight')
        plt.close()

        correlation_results.append({
            'comparison_type': label,
            'eeg_metric': y_col,
            'perplexity_metric': x_col,
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'kendall_r': kendall_r,
            'kendall_p': kendall_p
        })

    for perp_metric in perplexity_metrics:
        safe_perp_name = perp_metric.replace(' ', '_').lower()
        for eeg_metric in eeg_metrics:
            plot_and_corr(
                x_col=f'clean_{perp_metric}',
                y_col=f'clean_{eeg_metric}_eeg',
                label='Clean',
                color='blue',
                fname_prefix=f'clean_{eeg_metric}_eeg_vs_{safe_perp_name}'
            )
            plot_and_corr(
                x_col=f'obf_{perp_metric}',
                y_col=f'obf_{eeg_metric}_eeg',
                label='Obf',
                color='red',
                fname_prefix=f'obf_{eeg_metric}_eeg_vs_{safe_perp_name}'
            )

    correlation_df = pd.DataFrame(correlation_results)
    # Save correlation statistics to CSV
    if output_dir_scatter_plots:
        correlation_csv_path = os.path.join(output_dir_scatter_plots, "correlation_statistics.csv")
        correlation_df.to_csv(correlation_csv_path, index=False)
        print(f"Correlation statistics saved to {correlation_csv_path}")

    return correlation_df


def split_results_by_aoi_region_in_memory(results, optimal_aoi):
    """
    For each file, return a dict with only the optimal AOI region (in memory).
    Args:
        results: dict of file results
        optimal_aoi: dict mapping file_name to list of optimal AOI types
    Returns:
        dict mapping file_path with AOI type to file results
    """
    split_results = {}
    for file_path, file_data in results.items():
        file_name = os.path.basename(file_path).split('.')[0]
        if file_name in optimal_aoi:
            for aoi_type in optimal_aoi[file_name]:
                new_data = file_data.copy()
                if 'aoi_regions' in new_data and aoi_type in new_data['aoi_regions']:
                    new_data['aoi_regions'] = {aoi_type: new_data['aoi_regions'][aoi_type]}
                    new_data['aoi_labels'] = {aoi_type: new_data['aoi_labels'][aoi_type]}
                    split_results[file_path + f"-{aoi_type}"] = new_data
    return split_results


def analyze_individual_snippets(results, java_files_dir_with_aois, output_dir=None, averaged_trials=None,
                                std_trials=None, t_min=0.4, t_max=0.8):
    """
    Analyze all code snippets

    Args:
        results: Dictionary of file results
        output_dir: Optional directory to save figures

    """
    # read out optimal aoi region for each file
    df = pd.read_csv(os.path.join(java_files_dir_with_aois, "aoi_sizes.csv"), delimiter=";")
    # Filter rows where OPTIMAL is True
    optimal_aoi_df = df[df['OPTIMAL']]
    # look which aoi region is optimal for each file
    optimal_aoi = {}
    for index, row in optimal_aoi_df.iterrows():
        file_name = row['Snippet']
        aoi_type = str(row['NO'])
        if file_name in optimal_aoi:
            optimal_aoi[file_name].append(aoi_type)
        else:
            optimal_aoi[file_name] = [aoi_type]

    results = split_results_by_aoi_region_in_memory(results, optimal_aoi)

    for file in results:
        file_name = file.rsplit('-', 1)[0] + f"-AOI {file.rsplit('-', 1)[1]}"
        print(f"Analyzing file: {file_name}")

        # Generate output path if directory is provided
        output_path = None
        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_path = os.path.join(output_dir, f"{file_name}_analysis.pdf")

        if averaged_trials is not None and std_trials is not None:
            # Get the EEG signals for the clean and obfuscated files
            eeg_signal = averaged_trials[file_name][0] if file_name in averaged_trials else None
            eeg_std = std_trials[file_name][0] if file_name in std_trials else None
            if file_name not in averaged_trials:
                print(f"Skipped file: {file_name}, No EEG Signal found")
            fig = plot_token_perplexity_for_individual_snippet(results, file, output_path, t_min=t_min,
                                                               t_max=t_max, eeg_signal=eeg_signal,
                                                               eeg_std=eeg_std)
        else:
            # Compare token perplexity for this pair
            fig = plot_token_perplexity_for_individual_snippet(results, file, output_path)

        plt.close(fig)

    return


def create_eeg_perplexity_correlation_analysis_df_individual_snippets(results_path, output_dir=None,
                                                                      averaged_trials=None):
    """
    Analyzes EEG and perplexity correlations at the individual snippet/AOI level.

    Each row in the output corresponds to a single snippet or AOI,
    without pairing or clean/obfuscated distinctions.

    Args:
        results_path: Path to the perplexity results CSV
        output_dir: Directory to save analysis CSV
    """
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load perplexity data
    perplexity_df = pd.read_csv(results_path)

    analysis_data = []

    for _, row in perplexity_df.iterrows():
        snippet_file = row['File']
        # change snippet name from "13-clean-v1-1" to "13-clean-v1-AOI 1"
        snippet_name = snippet_file.rsplit('-', 1)[0] + f"-AOI {snippet_file.rsplit('-', 1)[1]}"

        mean_eeg = np.mean(averaged_trials[snippet_name][0]) if snippet_name in averaged_trials else None
        max_eeg = np.max(averaged_trials[snippet_name][0]) if snippet_name in averaged_trials else None
        aoi_label_entry = row['AOI Label']
        aoi_dict = ast.literal_eval(aoi_label_entry)
        # Get the first (and only) key
        first_key = next(iter(aoi_dict))
        # Extract the label
        if not aoi_dict or first_key not in aoi_dict:
            print(f"Warning: AOI label for {snippet_file} is missing or empty.")
            category = "No AOI"
        else:
            category = aoi_dict[first_key]['category']
        entry = {
            'file': snippet_file,
            'mean_eeg': mean_eeg,
            'max_eeg': max_eeg,
            'aoi_category': category,
        }

        # Add perplexity metrics
        for metric in perplexity_metrics:
            entry[metric] = row[metric]

        analysis_data.append(entry)

    analysis_df = pd.DataFrame(analysis_data)

    if output_dir:
        output_path = os.path.join(output_dir, "eeg_perplexity_analysis_individual.csv")
        # delete the file if it exists
        if os.path.exists(output_path):
            os.remove(output_path)
        analysis_df.to_csv(output_path, index=False)
        print(f"Saved individual EEG-perplexity analysis to: {output_path}")

    return analysis_df
