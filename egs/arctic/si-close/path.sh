export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda
export PRJ_ROOT=../../..
source $PRJ_ROOT/tools/venv/bin/activate
export PATH=$PATH:$PRJ_ROOT/src/bin:$PRJ_ROOT/src/utils
export PYTHONPATH=$PRJ_ROOT/src/nets:$PRJ_ROOT/src/utils
