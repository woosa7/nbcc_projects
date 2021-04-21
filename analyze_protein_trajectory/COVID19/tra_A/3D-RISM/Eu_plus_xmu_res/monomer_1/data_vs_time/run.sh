
# set dirname

dirname="monomer_1"

# make list_times and list_files

cp ../../../xmu_decomposition/data_vs_time/list_times .

ls ../../../../anal_renergy/$dirname/renergy_*.dat        > list_files_renergy
ls ../../../xmu_decomposition/data/*/$dirname/xmu_res.dat > list_files_xmu_res

# perform analysis

/homes/eta/users/chong/project_covid19/tools/calc_Eu_plus_xmu_res/for_BINARY_and_monomer_1/run.exe

# delete unnecessary file

rm -rf list_times
rm -rf list_files_renergy
rm -rf list_files_xmu_res

