
# set dirname

dirname="BINARY"

# make directory for dirname if necessary

if [ ! -d $dirname ]; then
  mkdir $dirname
fi

# cd to dirname and copy list_times

cd $dirname
cp ../list_times .

# make list_files (NOTICE: list_times must be prepared by hand)

ls ../../data/*/$dirname/protein.xmu > list_files

# perform analysis

/homes/eta/users/chong/project_covid19/tools/analytic_xmu_vs_time/run.exe

# delete unnecessary files

rm -rf list_times
rm -rf list_files

