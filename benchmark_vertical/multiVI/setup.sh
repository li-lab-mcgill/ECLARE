module load StdEnv/2020
module load python/3.10.2
python -m venv /home/dmannk/projects/def-liyue/dmannk/CLARE/benchmark_vertical/multiVI/scvi_env
source /home/dmannk/projects/def-liyue/dmannk/CLARE/benchmark_vertical/multiVI/scvi_env/bin/activate

module load scipy-stack/2023b

pip install --upgrade pip

pip install --no-deps scvi-tools

pip install torch
pip install -U typing-extensions
pip install sympy
pip install lightning
pip install rich
pip install anndata
pip install mudata
pip install ml_collections
pip install docrep
pip install jax==0.4.21 jaxlib==0.4.19 ml-dtypes==0.2.0
pip install --no-deps flax==0.8.1
pip install pyro-ppl
pip install numpyro
pip install scikit-learn
pip install sparse
pip install xarray

#pip install scanpy
pip install scib-metrics==0.4.1
pip install pybedtools pybiomart celltypist
pip install muon
