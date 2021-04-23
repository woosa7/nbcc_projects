
# main part

cat list_files | while read filename
do

  echo $filename

  cd data

  mkdir $filename
  cd $filename

  # copy necessary files

  cp ../../../../set_residue_info_BINARY/BINARY_natom.dat .
  cp ../../../../set_residue_info_BINARY/residue_info_BINARY.dat ./residue_info.dat

  # copy pdb file

  cp ../../../pdb_from_crd/"$filename".pdb ./protein.pdb

  # perform analysis

  /homes/eta/users/chong/project_covid19/tools/find_inter_monomer_heavy_atom_contacts/run.exe

  # delete unnecessary files

  rm -rf BINARY_natom.dat
  rm -rf residue_info.dat

  rm -rf protein.pdb

  # for the next step ...

  cd ../..

done

