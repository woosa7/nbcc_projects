
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

ls ../thma_data/*/protein.thm > list_files_thm

ls ../thma_data/*/xmu_tot_from_xmua.dat > list_files_xmu_tot_from_xmua
ls ../thma_data/*/xmu_res.dat           > list_files_xmu_res

ls ../thma_data/*/ene_tot_from_enea.dat > list_files_ene_tot_from_enea
ls ../thma_data/*/ene_res.dat           > list_files_ene_res

ls ../thma_data/*/ent_tot_from_enta.dat > list_files_ent_tot_from_enta
ls ../thma_data/*/ent_res.dat           > list_files_ent_res

ls ../thma_data/*/euv_tot_from_euva.dat > list_files_euv_tot_from_euva
ls ../thma_data/*/euv_res.dat           > list_files_euv_res

ls ../thma_data/*/evv_tot_from_evva.dat > list_files_evv_tot_from_evva
ls ../thma_data/*/evv_res.dat           > list_files_evv_res

#
# perform the analysis
#

/homes/epsilon/users/nbcc/HP36_tutorial/tools/thm_vs_time/run.exe

/homes/epsilon/users/nbcc/HP36_tutorial/tools/xmu_vs_time/run.exe << !
$nres
!

/homes/epsilon/users/nbcc/HP36_tutorial/tools/ene_vs_time/run.exe << !
$nres
!

/homes/epsilon/users/nbcc/HP36_tutorial/tools/ent_vs_time/run.exe << !
$nres
!

/homes/epsilon/users/nbcc/HP36_tutorial/tools/euv_vs_time/run.exe << !
$nres
!

/homes/epsilon/users/nbcc/HP36_tutorial/tools/evv_vs_time/run.exe << !
$nres
!

#
# move into directory
#

if [ ! -d xmu_data ]; then
  mkdir xmu_data
fi

if [ ! -d ene_data ]; then
  mkdir ene_data
fi

if [ ! -d ent_data ]; then
  mkdir ent_data
fi

if [ ! -d euv_data ]; then
  mkdir euv_data
fi

if [ ! -d evv_data ]; then
  mkdir evv_data
fi

if [ ! -d pmv_data ]; then
  mkdir pmv_data
fi

mv xmu_*.dat xmu_data
mv ene_*.dat ene_data
mv ent_*.dat ent_data
mv euv_*.dat euv_data
mv evv_*.dat evv_data
mv pmv_*.dat pmv_data

#
# delete unnecessary files
#

rm -rf list_files_*

