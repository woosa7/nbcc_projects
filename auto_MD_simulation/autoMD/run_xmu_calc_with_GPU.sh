# set GPU device number (either 0 or 1)
GPU_device_num=$1
echo $GPU_device_num

# do xmu calculations for each pre_rsm file in list_files
cat list_files | while read filename

do
  echo $filename
  cd execute

  # copy protein.pql & protein.pdb
  cp ../../../2_pql/protein.pql .
  cp ../../4_extract_pdb/pdb/"$filename" ./protein.pdb

  # create pdb and temp_protein.inp files from pre_rsm.dat
  /homes/epsilon/users/nbcc/HP36_tutorial/tools/mkrsm_from_pql_and_pdb/mkrsm_monomer/run.exe

  # create protein.inp
  cp /opt/nbcc/common/3D-RISM/header_m_BX_128_NX_128.inp ./protein.inp
  cat temp_protein.inp >> protein.inp

  # copy solvent input file
  cp /opt/nbcc/common/3D-RISM/tip3p_combined_300K.xsv ./solvent.xsv

  # perform 3D-RISM calculation
  rism3d-x protein.inp $GPU_device_num > /dev/null

  # copy xmu data
  cp protein.xmu ../xmu_data/$filename.xmu

  # delete unnecessary files
  rm -rf temp_protein.inp
  rm -rf solvent.xsv
  rm -rf protein.pql
  rm -rf protein.pdb
  rm -rf protein.inp
  rm -rf protein.xmu
  cd ..
done
