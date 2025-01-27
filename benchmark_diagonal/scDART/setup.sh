python3.9 -m venv scdart_env

git clone https://github.com/PeterZZQ/scDART.git
cd /Users/dmannk/cisformer/CLARE/benchmark_vertical/scDART/scDART
pip install .

pip install ipykernel
python -m ipykernel install --user --name=scdart_env --display-name "Python (scdart_env)"

pip install numpy==1.25.2

pip install mudata==0.2.3
pip install anndata==0.10.3

pip install scanpy
pip install pybedtools pybiomart celltypist
pip install muon

pip install scib-metrics==0.4.1
pip install jax==0.4.13
pip install jaxlib==0.4.13
pip install ml-dtypes==0.1.0

pip install graphtools==1.5.3
pip install phate