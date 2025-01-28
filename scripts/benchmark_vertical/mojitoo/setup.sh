module load StdEnv/2023
python -m venv /home/dmannk/projects/def-liyue/dmannk/CLARE/benchmark_vertical/mojitoo/mojitoo_env
source /home/dmannk/projects/def-liyue/dmannk/CLARE/benchmark_vertical/mojitoo/mojitoo_env/bin/activate

pip install --upgrade pip

git clone https://github.com/CostaLab/MOJITOO.git
cd MOJITOO/pymojitoo
pip install .

pip uninstall anndata mudata
pip install anndata==0.10.3
pip install mudata==0.2.3

pip uninstall numpy
pip install numpy==1.26.4

pip install pybedtools pybiomart celltypist torch
pip install episcanpy==0.4.0
pip install muon

pip uninstall scib-metrics jax jaxlib ml-dtypes
pip install scib-metrics==0.4.1
pip install jax==0.4.13
pip install jaxlib==0.4.13
pip install ml-dtypes==0.1.0