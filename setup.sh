#!/bin/bash

export GPGPUSIM_ROOT="/home/vlad/classes/cmpe214/gpgpu-sim_distribution-master"
export GPGPUSIM_CUDART_LIB=$GPGPUSIM_ROOT/lib/gcc-5.3.1/cuda-11050/release
export LD_LIBRARY_PATH=$GPGPUSIM_CUDART_LIB
export CUDA_INSTALL_PATH=/usr/local/cuda-11.5
export CUDAHOME=$CUDA_INSTALL_PATH
export SDKINCDIR=/home/vlad/classes/cmpe214/lab5/cuda-workshop/cutil/inc

