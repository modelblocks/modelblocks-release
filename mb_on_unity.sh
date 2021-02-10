# make tmp directory (necessary for certain Unix commands)
mkdir $HOME/tmp

# initialize and activate new conda env (need to be on (base) environment of miniconda3)
conda create -n mb
#conda activate mb
source activate mb #ej - this didn't work for me so i changed it to 'conda activate'

# install compilers/BLAS dependencies (for armadillo), python/scipy (for NN training)
# python >= 3.5 recommended
# NOTE: tensorflow<2.0 because of CDR
conda install -c conda-forge compilers libblas liblapack arpack superlu mkl python scipy pandas librosa r-base r-lme4 r-languageR r-optimx r-ggplot2 r-optparse tensorflow"<2.0" pytorch-gpu torchvision cudatoolkit=10.1 -c pytorch

# install pytorch (for NN training) - Unity node 070 has cuda driver 10.1
#conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

## install r
#conda install r-base r-lme4 r-languageR r-optimx r-ggplot2 r-optparse

# install pymvpa2
#conda install -c conda-forge librosa
# pymvpa2

# install up-to-date compilers that can use c++17
conda install -c conda-forge compilers armadillo

## download armadillo from source
#wget http://sourceforge.net/projects/arma/files/armadillo-9.900.3.tar.xz
#tar -xJf armadillo-9.900.3.tar.xz

## install armadillo to conda env, using mkl as primary BLAS
#cd armadillo-9.900.3
#cmake -D DETECT_HDF5=false . -DCMAKE_INSTALL_PREFIX:PATH=$CONDA_PREFIX
#make
#make install
