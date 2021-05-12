#!/usr/bin/env bash

if [ -z ${PYTHON+x} ]; then
  export PYTHON="$HOME/anaconda3/envs/prowave/bin/python";
fi

if [ -z ${WORKSPACE+x} ]; then
  export WORKSPACE="$HOME/workspace"
fi

export CUDA_VISIBLE_DEVICES=$3
$PYTHON -m utils.run_prep $WORKSPACE/$1/$2
export CUDA_VISIBLE_DEVICES=$3
$PYTHON -m utils.run_min $WORKSPACE/$1/$2
export CUDA_VISIBLE_DEVICES=$3
$PYTHON -m utils.run_eq $WORKSPACE/$1/$2
export CUDA_VISIBLE_DEVICES=$3
$PYTHON -m utils.run_md $WORKSPACE/$1/$2 1

unset WORKSPACE
unset PYTHON
