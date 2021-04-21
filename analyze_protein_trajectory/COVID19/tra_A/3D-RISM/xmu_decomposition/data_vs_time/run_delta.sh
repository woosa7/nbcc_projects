
# set dirname

dirname="delta"

# set nres

nres=648

# make directory for dirname if necessary

if [ ! -d $dirname ]; then
  mkdir $dirname
fi

# cd to dirname and copy list_times (NOTICE: list_times must be prepared by hand)

cd $dirname

cp ../list_times .
cp ../list_times ./list_times_decom

# make list_files

ls ../../data/*/BINARY/protein.xmu    > list_files_xmu_BINARY
ls ../../data/*/monomer_1/protein.xmu > list_files_xmu_monomer_1
ls ../../data/*/monomer_2/protein.xmu > list_files_xmu_monomer_2

ls ../../data/*/$dirname/delta_xmu_tot_from_xmua.dat > list_files_delta_xmu_tot_from_xmua
ls ../../data/*/$dirname/delta_xmu_res.dat           > list_files_delta_xmu_res

# perform analysis

/homes/eta/users/chong/project_covid19/tools/delta_analytic_xmu_vs_time/run.exe

/homes/eta/users/chong/project_covid19/tools/delta_xmu_vs_time/run.exe << !
$nres
!

# delete unnecessary files

rm -rf list_times
rm -rf list_times_decom

rm -rf list_files*

# go back to the original directory

cd ..

