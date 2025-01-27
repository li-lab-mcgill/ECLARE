python -m venv /home/dmannk/projects/def-liyue/dmannk/CLARE/benchmark_vertical/glue/glue_env
source /home/dmannk/projects/def-liyue/dmannk/CLARE/benchmark_vertical/glue/glue_env/bin/activate

pip install --upgrade pip

pip install numpy==1.25.2

pip install scglue

pip install mudata==0.2.3
pip install anndata==0.10.3

pip install pybedtools pybiomart celltypist
pip install muon
pip install episcanpy
pip install scikit-misc

pip install scib-metrics==0.4.1
pip install jax==0.4.13
pip install jaxlib==0.4.13
pip install ml-dtypes==0.1.0

pip install tensorboardX

pip install scipy==1.11.2