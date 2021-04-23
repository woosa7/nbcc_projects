
# copy BINARY_nres.dat

cp ../../../set_residue_info_BINARY/BINARY_nres.dat .

# make list_files

ls ../data/*/SB_HB_residues.dat > list_files

# perform analysis

/homes/eta/users/chong/project_covid19/tools/analysis_inter_monomer_SB_HB/population_SB_HB/run.exe

# delete unnecessary file(s)

rm -rf BINARY_nres.dat

