#####################################################
# Following are the dependencies and the versions  
# used in the LLM predictivity project
#####################################################

# go to the (base) environment
conda deavtivate

# create conda env
conda create -n llm-pred python=3.7

# activate conda env
conda activate llm-pred

# install pytorch v1.13.0
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
# conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia

# install huggingface transformer v4.24.0
pip install transformers==4.24.0