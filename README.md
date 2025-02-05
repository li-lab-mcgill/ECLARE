# `ECLARE`: multi-teacher contrastive learning via ensemble distillation for diagonal integration of single-cell multi-omic data

This repository is dedicated to <ins>**E**</ins>nsemble knowledge distillation for <ins>**C**</ins>ontrastive <ins>**L**</ins>earning of <ins>**A**</ins>TAC and <ins>**R**</ins>NA <ins>**E**</ins>mbeddings, a.k.a. `ECLARE` :zap::cake:.

The manuscript is currently available on [bioRxiv](https://doi.org/10.1101/2025.01.24.634799).

<details>
<summary>Installation</summary>

1. First, clone the repository:

    ```bash
    git clone https://github.com/li-lab-mcgill/ECLARE.git
    cd ECLARE
    ```

2. Create a virtual environment (use Python 3.9.6 for best reproducibility):
    ```bash
    python -m venv eclare_env
    ```

3. Activate the virtual environment
    
    Windows
    ```bash
    eclare_env\Scripts\activate
    ```
    
    macOS and Linux
    ```bash 
    source eclare_env/bin/activate
    ```

    Git Bash on Windows
    ```bash
    source eclare_env/Scripts/activate
    ```

4. Install the package:
    For standard installation:
    ```bash
    pip install .
    ```

    For editable installation (recommended for development):
    ```bash
    pip install -e .
    ```
</details>

<details>
<summary>Configuration</summary>

Before running the application, you need to set up your configuration file. Follow these steps:

1. Copy the template configuration file:

    ```bash
    cp config/config_template.yaml config/config.yaml
    ```

2. Edit `config.yaml` to suit your environment. Update paths and settings as necessary:

    ```yaml
    active_environment: "local_directories"

    local_directories:
      outpath: "/your/custom/output/path"
      datapath: "/your/custom/data/path"
    ```
</details>

<details>
<summary>Requirements</summary>

- Python ≥ 3.9 (3.9.6 for best reproducibility)
- See `setup.py` for a complete list of dependencies
</details>

<details>
<summary>Overview of ECLARE framework</summary>

---



ECLARE (Ensemble knowledge distillation for Contrastive Learning of ATAC and RNA Embeddings) is a framework designed to integrate single-cell multi-omic data, specifically scRNA-seq and scATAC-seq data, through these key components:

1. **Multi-Teacher Knowledge Distillation**:
   - Multiple teacher models are trained on paired datasets (where RNA and ATAC data are available for the same cells)
   - These teachers then guide a student model that works with unpaired data
   - This approach helps transfer knowledge from well-understood paired samples to situations where only unpaired data is available

2. **Contrastive Learning**:
   - Uses a refined contrastive learning objective to learn representations of both RNA and ATAC data
   - Helps align features across different modalities (RNA and ATAC)
   - Enables the model to understand relationships between different data types

3. **Transport-based Loss**:
   - Implements a transport-based loss function for precise alignment between RNA and ATAC modalities
   - Helps ensure that the learned representations are biologically meaningful

The framework is particularly valuable because it:
- Addresses the common problem of limited paired multi-omic data
- Enables integration of unpaired data through knowledge transfer
- Preserves biological structure in the integrated data
- Facilitates downstream analyses like gene regulatory network inference

---

Figure 1 from manuscript: Overview of ECLARE

<div style="display: flex; justify-content: center; margin: 20px;">
  <div style="background: white; padding: 20px; border-radius: 8px;">
    <img src="fig1_landscape_no_alpha.png" alt="ECLARE Framework"/>
  </div>
</div>
</details>

<details>
<summary>Demo: analysis on sample paired datasets</summary>
We provide a demo notebook to analyze the sample paired datasets. This notebook is located in `scripts/sample_paired_datasets_analysis.ipynb`.

This analysis is based on using DLPFC_Anderson and DLPFC_Ma as source datasets and PFC_Zhu as target dataset. See Table 1 in the manuscript for more details about datasets.

Sample data is available from Zenodo at https://doi.org/10.5281/zenodo.14794845. Instructions for downloading the data are available in the notebook.
</details>

