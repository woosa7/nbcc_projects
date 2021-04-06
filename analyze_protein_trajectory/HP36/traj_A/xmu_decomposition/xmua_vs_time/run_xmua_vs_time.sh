
#
# set nres
#

nres=36

#
# make necessary lists
#
#   NOTICE : (1) list_times is not created here, and you should make it by hand
#            (2) check whether the number of data in list_times is consistent with those in other lists
#

ls ../xmua_data/*/xmu_tot_from_xmua.dat > list_files_xmu_tot_from_xmua
ls ../xmua_data/*/xmu_res.dat           > list_files_xmu_res

#
# perform the analysis
#

/homes/epsilon/users/nbcc/HP36_tutorial/tools/xmu_vs_time/run.exe << !
$nres
!

#
# delete unnecessary files
#

rm -rf list_files_*

