# `ECLARE`: multi-teacher contrastive learning via ensemble distillation for diagonal integration of single-cell multi-omic data

This repository is dedicated to <ins>**E**</ins>nsemble knowledge distillation for <ins>**C**</ins>ontrastive <ins>**L**</ins>earning of <ins>**A**</ins>TAC and <ins>**R**</ins>NA <ins>**E**</ins>mbeddings, a.k.a. `ECLARE` :zap::cake:.

## Installation

It is highly recommended to install ECLARE within a virtual environment to avoid dependency conflicts.

1. First, clone the repository:

    ```bash
    git clone https://github.com/li-lab-mcgill/ECLARE.git
    cd ECLARE
    ```

2. Create and activate a virtual environment:
    Create virtual environment
    ```bash
    python -m venv eclare_env
    ```

3. Activate virtual environment
    
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

### Configuration

Before running the application, you need to set up your configuration file. Follow these steps:

1. Copy the template configuration file:

    ```bash
    cp config/config_template.yaml config.yaml
    ```

2. Edit `config.yaml` to suit your environment. Update paths and settings as necessary:

    ```yaml
    active_environment: "local_directories"

    local_directories:
      outpath: "/your/custom/output/path"
      datapath: "/your/custom/data/path"
    ```

### Requirements
- Python ≥ 3.9
- See `setup.py` for a complete list of dependencies
