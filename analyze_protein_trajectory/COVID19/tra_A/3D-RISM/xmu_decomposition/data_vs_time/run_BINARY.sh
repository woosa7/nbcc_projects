
# set dirname

dirname="BINARY"

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

ls ../../data/*/$dirname/protein.xmu > list_files

ls ../../data/*/$dirname/xmu_tot_from_xmua.dat > list_files_xmu_tot_from_xmua
ls ../../data/*/$dirname/xmu_res.dat           > list_files_xmu_res

# perform analysis

/homes/eta/users/chong/project_covid19/tools/analytic_xmu_vs_time/run.exe

/homes/eta/users/chong/project_covid19/tools/xmu_vs_time/run.exe << !
$nres
!

# delete unnecessary files

rm -rf list_times
rm -rf list_times_decom

rm -rf list_files*

# go back to the original directory

cd ..

