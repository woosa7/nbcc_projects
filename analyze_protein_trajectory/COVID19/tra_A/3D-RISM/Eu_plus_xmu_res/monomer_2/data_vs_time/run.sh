
# set dirname

dirname="monomer_2"

# make list_times and list_files

cp ../../../xmu_decomposition/data_vs_time/list_times .

ls ../../../../anal_renergy/$dirname/renergy_*.dat        > list_files_renergy
ls ../../../xmu_decomposition/data/*/$dirname/xmu_res.dat > list_files_xmu_res

# perform analysis (special treatment for monomer_2 in BINARY complex)

/homes/eta/users/chong/project_covid19/tools/calc_Eu_plus_xmu_res/for_monomer_2_in_BINARY_complex/run.exe

# delete unnecessary file

rm -rf list_times
rm -rf list_files_renergy
rm -rf list_files_xmu_res

