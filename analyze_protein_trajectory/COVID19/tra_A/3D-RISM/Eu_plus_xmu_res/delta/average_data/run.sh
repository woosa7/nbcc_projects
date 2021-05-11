
# set nres

nres=648

# make list_files

rm -rf list_files
touch list_files

ls ../data_vs_time/delta_Eu_plus_xmu_res_vs_time.dat >> list_files

# perform analysis

/homes/eta/users/chong/project_covid19/tools/average_Eu_plus_xmu_vs_residue/for_data_vs_time_with_std/run.exe << !
$nres
!

mv average_Eu_plus_xmu_vs_residue.dat delta_Eu_plus_xmu_vs_residue.dat 

