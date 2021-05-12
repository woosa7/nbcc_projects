cp ../../set_residue_info_BINARY/BINARY_nres.dat .

dat_dir="../data/"

python population_SB_HB.py $dat_dir

# rm -rf BINARY_nres.dat
