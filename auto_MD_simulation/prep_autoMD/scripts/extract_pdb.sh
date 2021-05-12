#!/usr/bin/env bash

if [ -z ${PYTHON+x} ]; then
  export PYTHON="$HOME/anaconda3/envs/prowave/bin/python";
fi

if [ -z ${WORKSPACE+x} ]; then
  export WORKSPACE="$HOME/workspace"
fi

$PYTHON -m utils.extract_pdb $WORKSPACE/$1/$2

unset PYTHON
unset WORKSPACE