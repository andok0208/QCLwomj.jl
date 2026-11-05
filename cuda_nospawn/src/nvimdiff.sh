#!/bin/bash

src=$1
#refdir=../../../wig_rev/spinbo1d/julia/cuda/gpuctx/
#refdir=../src/
#refdir=../../../wig_new/cuda_spawn/src_test/
refdir=../../../qcltfd/cuda/src_test/

#nvim -d ./"$src"  "$refdir"/"$src"
vimdiff ./"$src"  "$refdir"/"$src"

