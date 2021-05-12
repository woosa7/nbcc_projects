#!/usr/bin/env bash

if [ -z ${PDBLIST+x} ]; then
  export PDBLIST="$HOME";
fi

if [ -z ${WORKSPACE+x} ]; then
  export WORKSPACE="$HOME/workspace"
fi

while IFS='' read -r line || [[ -n "$line" ]]; do
    echo $line
    sbatch --output /dev/null --error /dev/null ./utils/scripts/extract_pdb.sh $1 $line
done < "$PDBLIST/$1.txt"

unset PDBLIST
unset WORKSPACE
