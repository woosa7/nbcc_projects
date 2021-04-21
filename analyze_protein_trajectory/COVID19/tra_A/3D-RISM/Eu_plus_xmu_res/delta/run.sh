
dir_set_residue_info="/homes/eta/users/chong/project_covid19/CV30_RBD/set_residue_info_BINARY"

dir_BINARY="../BINARY"
dir_monomer_1="../monomer_1"
dir_monomer_2="../monomer_2"

# copy necessary files

cp $dir_set_residue_info/residue_info_BINARY.dat .
cp $dir_set_residue_info/residue_info_monomer_1.dat .
cp $dir_set_residue_info/residue_info_monomer_2.dat .

cp $dir_BINARY/average_data/average_Eu_plus_xmu_vs_residue.dat    ./average_Eu_plus_xmu_vs_residue_BINARY.dat
cp $dir_monomer_1/average_data/average_Eu_plus_xmu_vs_residue.dat ./average_Eu_plus_xmu_vs_residue_monomer_1.dat
cp $dir_monomer_2/average_data/average_Eu_plus_xmu_vs_residue.dat ./average_Eu_plus_xmu_vs_residue_monomer_2.dat

# perform analysis

/homes/eta/users/chong/project_covid19/tools/diff_Eu_plus_xmu_vs_residue/for_BINARY_simulations/run.exe

# delete unnecessary files

rm -rf average_Eu_plus_xmu_vs_residue_*.dat
rm -rf residue_info_*.dat

