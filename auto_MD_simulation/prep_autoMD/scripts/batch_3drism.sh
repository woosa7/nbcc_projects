#!/usr/bin/env bash

if [ -z ${PDBLIST+x} ]; then
  export PDBLIST="$HOME";
fi

if [ -z ${WORKSPACE+x} ]; then
  export WORKSPACE="$HOME/workspace"
fi

while IFS='' read -r line || [[ -n "$line" ]]; do
    echo $line
    sbatch --output $WORKSPACE/$1/$line/out.log --error $WORKSPACE/$1/$line/error.log --gres gpu:1 ./utils/scripts/run_3drism.sh $1 $line 0
done < "$PDBLIST/$1.txt"

unset PDBLIST
unset WORKSPACE
