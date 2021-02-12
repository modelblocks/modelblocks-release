# initialize and activate new conda environment, assumes working installation of miniconda3
# compatible with TensorFlow 1.15 (https://github.com/tensorflow/models/tree/archive/research/lm_1b)
conda create -n jlstm_env python=3.7 bazel -c conda-forge

# installs TensorFlow from pip
CONDA_BASE=$(conda info --base)
$CONDA_BASE/envs/jlstm_env/bin/pip install tensorflow==1.15
