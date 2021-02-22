# initialize and activate new conda environment, assumes working installation of miniconda3
# compatible with PyTorch 1.0.0 (https://github.com/facebookresearch/colorlessgreenRNNs)
conda create -n glstm_env python=3.5 pytorch==1.0.0 torchvision==0.2.1 cuda100 -c pytorch pandas
