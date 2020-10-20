# make tmp directory (necessary for certain Unix commands)
mkdir $HOME/tmp

# initialize and activate new conda env (need to be on (base) environment of miniconda3)
conda create -n mb
source activate mb

# install compilers/BLAS dependencies (for armadillo), python/scipy (for NN training)
# python >= 3.5 recommended
conda install -c conda-forge compilers libblas liblapack arpack superlu mkl python scipy pandas

# install pytorch (for NN training) - Unity node 070 has cuda driver 10.1
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

# install up-to-date compilers that can use c++17
conda install compilers

# download armadillo from source
wget http://sourceforge.net/projects/arma/files/armadillo-9.900.3.tar.xz
tar -xJf armadillo-9.900.3.tar.xz

# install armadillo to conda env, using mkl as primary BLAS
cd armadillo-9.900.3
cmake -D DETECT_HDF5=false . -DCMAKE_INSTALL_PREFIX:PATH=$CONDA_PREFIX
make
make install
