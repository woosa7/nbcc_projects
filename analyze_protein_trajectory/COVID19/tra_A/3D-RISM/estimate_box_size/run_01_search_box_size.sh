
cat list_files | while read filename
do

  cp ../../pdb_from_rst/$filename ./protein.pdb
  /homes/eta/users/chong/project_covid19/tools/estimate_box_size/run.exe
  rm -rf protein.pdb

done
