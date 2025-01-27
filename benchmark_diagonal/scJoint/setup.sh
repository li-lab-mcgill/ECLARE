python3.9 -m venv /home/dmannk/projects/def-liyue/dmannk/CLARE/benchmark_diagonal/scJoint/scjoint_env
source /home/dmannk/projects/def-liyue/dmannk/CLARE/benchmark_vertical/scJoint/scjoint_env/bin/activate

git clone https://github.com/SydneyBioX/scJoint.git

pip install ipykernel
python -m ipykernel install --user --name=scjoint_env --display-name "Python (scjoint_env)"

pip install numpy==1.25.2

pip install mudata==0.2.3
pip install anndata==0.10.3

pip install pybedtools pybiomart celltypist
pip install muon

pip install torch
pip install scikit-learn
pip install scanpy

pip install scib-metrics==0.4.1
pip install jax==0.4.13
pip install jaxlib==0.4.13
pip install ml-dtypes==0.1.0
