# initialize and activate new conda environment, assumes working installation of miniconda3
# due to frequent updates with both transformers and PyTorch, it may be advisable to install the dependencies manually
# https://huggingface.co/docs/transformers/en/installation
# https://pytorch.org
#conda create -n hf_env python=3.10
source activate hf_env
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip3 install transformers
