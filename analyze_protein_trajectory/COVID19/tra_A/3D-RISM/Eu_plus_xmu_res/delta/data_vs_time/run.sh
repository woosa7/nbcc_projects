
dir_set_residue_info="/homes/eta/users/chong/project_covid19/CV30_RBD/set_residue_info_BINARY"

# copy necessary files

cp $dir_set_residue_info/residue_info_BINARY.dat .
cp $dir_set_residue_info/residue_info_monomer_1.dat .
cp $dir_set_residue_info/residue_info_monomer_2.dat .

cp ../../BINARY/data_vs_time/Eu_plus_xmu_res_vs_time.dat    ./Eu_plus_xmu_res_vs_time_BINARY.dat
cp ../../monomer_1/data_vs_time/Eu_plus_xmu_res_vs_time.dat ./Eu_plus_xmu_res_vs_time_monomer_1.dat
cp ../../monomer_2/data_vs_time/Eu_plus_xmu_res_vs_time.dat ./Eu_plus_xmu_res_vs_time_monomer_2.dat

# perform analysis

/homes/eta/users/chong/project_covid19/tools/calc_delta_Eu_plus_xmu_res/run.exe

# delete unnecessary files

rm -rf residue_info_*.dat
rm -rf Eu_plus_xmu_res_vs_time_*.dat

