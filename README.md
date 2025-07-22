# Are Humans and LLMs Confused by the Same Code? An Empirical Study on Fixation-Related Potentials and LLM Perplexity

This repository contains the replication package for our ICSE 2026 paper submission:  
**"Are Humans and LLMs Confused by the Same Code? An Empirical Study on Fixation-Related Potentials and LLM Perplexity."**

We investigate the relationship between large language model (LLM) perplexity and human neurophysiological responses—specifically Fixation-Related Potentials (FRPs)—during program comprehension tasks. Our goal is to understand whether humans and LLMs exhibit confusion around the same code regions. The repository includes code, analysis notebooks, annotations, and plotting scripts to replicate all results and figures from the paper.

**Status**: Paper under review for ICSE 2026  

---

## Repository Structure

<details>
<summary><strong>Stimuli & Annotations</strong></summary>

* **[`code_snippets/`](code_snippets/)**
  Original Java code snippets from [Bergum et al.](https://arxiv.org/abs/2412.10099) — 72 snippets, each with three versions: `v0`, `v1`, and `v2`.

* **[`code_snippets_with_aois/`](code_snippets_with_aois/)**
  Same snippets, annotated with *human-defined confusing regions* (i.e., "atoms of confusion").

* **[`code_snippets_with_data_driven_aois/`](code_snippets_with_data_driven_aois/)**
  Snippets annotated using our *data-driven*, LLM-based approach to identify confusing code regions.

</details>

<details>
<summary><strong>Main Analysis Pipeline – <code>main/</code></strong></summary>

Includes Jupyter notebooks and helper scripts for all main analyses.

**RQ1 & RQ2: LLM Perplexity vs. FRPs (Human-Annotated AOIs)**

* [`rq1+2_perplexity_analysis.ipynb`](main/rq1+2_perplexity_analysis.ipynb)
  Main pipeline analyzing LLM perplexity and comparing with EEG FRPs.

**RQ3: Data-Driven AOIs via LLM Perplexity**

1. [`rq3.1_perplexity_algorithm.ipynb`](main/rq3.1_perplexity_algorithm.ipynb) – Detect confusing AOIs using LLM perplexity.
2. [`rq3.2_generate_snippets.ipynb`](main/rq3.2_generate_snippets.ipynb) – Map AOIs to screen position.
3. [`rq3.3_01d9_PRE_VIS_Special_Fixations.ipynb`](main/rq3.3_01d9_PRE_VIS_Special_Fixations.ipynb) – Extract fixations (adapted from [Bergum et al.](https://github.com/brains-on-code/AoC-FRP-Code)).
4. [`rq3.4_01g1_PRE_EEG_VIS_Set_FRP_markers.ipynb`](main/rq3.4_01g1_PRE_EEG_VIS_Set_FRP_markers.ipynb) – Set FRP markers based on AOIs. (adapted from [Bergum et al.](https://github.com/brains-on-code/AoC-FRP-Code))
5. [`rq3.5_perplexity_analysis_individual.ipynb`](main/rq3.5_perplexity_analysis_individual.ipynb) – Final Perplexity-FRP analysis for RQ3.

**Utility Files and Folders**

* [`algorithm_utils.py`](main/algorithm_utils.py), [`eeg_loader.py`](main/eeg_loader.py), [`utils.py`](main/utils.py), [`frp_utils/`](main/frp_utils/), [`fonts/`](main/fonts/), 
</details>

<details>
<summary><strong>Plotting Scripts – <code>paper_plots/</code></strong></summary>
Notebooks used to generate all visualizations and final paper figures:

* [`intro_plot.ipynb`](paper_plots/intro_plot.ipynb) – Overview figure
* [`rq1_plots.ipynb`](paper_plots/rq1_plots.ipynb) – Plots for RQ1
* [`rq2_plots.ipynb`](paper_plots/rq2_plots.ipynb) – Plots for RQ2
* [`rq3_plots.ipynb`](paper_plots/rq3_plots.ipynb) – Plots for RQ3

</details>

<details>
<summary><strong>Results – <code>model_results/</code></strong></summary>

* Contains all intermediate and final results generated during the experiments, organized by model.

</details>

> **Note:** The EEG and eye-tracking dataset used in this study is from [Bergum et al. (2025)](https://arxiv.org/abs/2412.10099) and is **not included** in this repository. Access can be requested directly from the original authors.

---

## Analysis Pipeline

<details>
<summary><strong>RQ1 & RQ2 – LLM Perplexity vs. FRPs (Human-Annotated AOIs)</strong></summary>

Run the following notebook to analyze:

* LLM token-level perplexity on the original code snippets.
* Corresponding Fixation-Related Potentials (FRPs) from EEG data.
* Alignment with human-defined regions of confusion (*atoms of confusion*).

[`main/rq1+2_perplexity_analysis.ipynb`](main/rq1+2_perplexity_analysis.ipynb)

</details>

<details>
<summary><strong>RQ3 – Data-Driven AOIs via LLM Perplexity</strong></summary>

This pipeline automatically identifies confusing code regions based on LLM perplexity and links them with FRP responses. Run the following notebooks **in sequence**:

1. **Detect regions of confusion via perplexity:**
   [`main/rq3.1_perplexity_algorithm.ipynb`](main/rq3.1_perplexity_algorithm.ipynb)

2. **Generate and map AOIs to screen positions:**
   [`main/rq3.2_generate_snippets.ipynb`](main/rq3.2_generate_snippets.ipynb)

3. **Extract fixations for identified AOIs:**
   [`main/rq3.3_01d9_PRE_VIS_Special_Fixations.ipynb`](main/rq3.3_01d9_PRE_VIS_Special_Fixations.ipynb)

4. **Set EEG markers at fixation points:**
   [`main/rq3.4_01g1_PRE_EEG_VIS_Set_FRP_markers.ipynb`](main/rq3.4_01g1_PRE_EEG_VIS_Set_FRP_markers.ipynb)

5. **Run final Perplexity-FRP analysis:**
   [`main/rq3.5_perplexity_analysis_individual.ipynb`](main/rq3.5_perplexity_analysis_individual.ipynb)

</details>

<details>
<summary><strong>Visualizations</strong></summary>
After completing the analysis, generate the corresponding visualizations using:

* [`paper_plots/intro_plot.ipynb`](paper_plots/intro_plot.ipynb)  # Overview figure
* [`paper_plots/rq1_plots.ipynb`](paper_plots/rq1_plots.ipynb)    # RQ1 results
* [`paper_plots/rq2_plots.ipynb`](paper_plots/rq2_plots.ipynb)    # RQ2 results
* [`paper_plots/rq3_plots.ipynb`](paper_plots/rq3_plots.ipynb)    # RQ3 results

</details>

---

## Acknowledgements

* EEG and eye-tracking data are from [Bergum et al. (2025)](https://arxiv.org/abs/2412.10099) — "Unexpected but informative: What fixation-related potentials tell us about the processing of confusing program code".
* Some scripts are adapted from the official [AoC-FRP-Code repository](https://github.com/brains-on-code/AoC-FRP-Code).

---

## License

This replication package is released for academic use only. Please contact the authors for other use cases.

---




