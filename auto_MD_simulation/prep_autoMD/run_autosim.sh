#!/usr/bin/env bash

yes|cp /homes/eta/users/aiteam/prowise/nbcc_autoMD/autoscript.py .
yes|cp /homes/eta/users/aiteam/prowise/nbcc_autoMD/md_prep_openmm.py .
yes|cp /homes/eta/users/aiteam/prowise/nbcc_autoMD/md_minimization.py .
yes|cp /homes/eta/users/aiteam/prowise/nbcc_autoMD/md_equilibration.py .
yes|cp /homes/eta/users/aiteam/prowise/nbcc_autoMD/md_production.py .
yes|cp /homes/eta/users/aiteam/prowise/nbcc_autoMD/md_calc_sfe.py .

for((i=0;i<300;i++));
do
  echo 'auto simulation' $i
  python autoscript.py md_parameters
done
