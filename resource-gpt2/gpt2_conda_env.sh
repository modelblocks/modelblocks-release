# initialize and activate new conda environment, assumes working installation of miniconda3
conda create -n gpt2_env python=3.6
CONDA_BASE=$(conda info --base)
pip=$CONDA_BASE/envs/gpt2_env/bin/pip
$pip install tensorflow==1.13.1
$pip install fire
$pip install regex
$pip install requests==2.21.0
$pip install tqdm==4.31.1
$pip install pandas

