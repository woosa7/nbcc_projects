
cat list_files | while read filename
do

echo $filename

cd data/$filename

#
# BINARY
#

  cd BINARY

  # copy residue_info.dat

  cp /homes/eta/users/chong/project_covid19/CV30_RBD/set_residue_info_BINARY/residue_info_BINARY.dat ./residue_info.dat

  # perform xmua analysis

  /homes/eta/users/chong/project_covid19/tools/xmu_analysis/run.exe

  # delete unnecessary files

  rm -rf residue_info.dat

  # for next

  cd ..

#
# monomer_1
#

  cd monomer_1

  # copy residue_info.dat

  cp /homes/eta/users/chong/project_covid19/CV30_RBD/set_residue_info_BINARY/residue_info_monomer_1.dat ./residue_info.dat

  # perform xmua analysis

  /homes/eta/users/chong/project_covid19/tools/xmu_analysis/run.exe

  # delete unnecessary files

  rm -rf residue_info.dat

  # for next

  cd ..

#
# monomer_2
#

  cd monomer_2

  # copy residue_info.dat

  cp /homes/eta/users/chong/project_covid19/CV30_RBD/set_residue_info_BINARY/residue_info_monomer_2.dat ./residue_info.dat

  # perform xmua analysis

  /homes/eta/users/chong/project_covid19/tools/xmu_analysis/run.exe

  # delete unnecessary files

  rm -rf residue_info.dat

  # for next

  cd ..

# 
# for the next step ...
#

cd ../..

done

