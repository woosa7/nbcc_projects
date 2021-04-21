export AMBERHOME=/opt/apps/amber20/amber20
export CUDA_HOME=/opt/cuda-10.0

export LD_LIBRARY_PATH=/usr/local/lib:$CUDA_HOME/lib64/:$CUDA_HOME/lib:$AMBERHOME/lib
export PATH=/usr/local/bin:/usr/bin:/bin:$AMBERHOME/bin:$CUDA_HOME/bin/:/opt/nbcc/bin:$HOME/bin

#
# set GPU device number (either 0 or 1)
#

GPU_device_num=0

#
# do xmu-decomposition calculations for each pdb file in list_files
#

cat list_files | while read filename
do

echo $filename

cd data

mkdir $filename
cd $filename

# copy protein.pql

cp ../../../../../set_pql_param/protein.pql .

# copy BINARY_natom.dat

cp ../../../../../set_residue_info_BINARY/BINARY_natom.dat .

# copy protein.pdb

cp ../../../../pdb_from_rst/$filename.pdb ./protein.pdb

# create temp.inp for BINARY, monomer_1 and monomer_2 
#
#   input:  BINARY_natom.dat
#           protein.pql (parameter only)
#           protein.pdb
#
#   output: temp_protein_BINARY.inp
#           temp_protein_monomer_1.inp
#           temp_protein_monomer_2.inp

/homes/eta/users/chong/project_covid19/tools/mkrsm_from_pql_and_pdb/mkrsm_BINARY_structure/run.exe

# make necessary directories

mkdir BINARY
mkdir monomer_1
mkdir monomer_2

# create protein.inp for BINARY

cp /homes/eta/users/chong/project_covid19/common/header_a_BX_160_NX_128.inp ./protein_BINARY.inp
cat temp_protein_BINARY.inp >> protein_BINARY.inp

mv protein_BINARY.inp ./BINARY/protein.inp

# create protein.inp for monomer_1

cp /homes/eta/users/chong/project_covid19/common/header_a_BX_160_NX_128.inp ./protein_monomer_1.inp
cat temp_protein_monomer_1.inp >> protein_monomer_1.inp

mv protein_monomer_1.inp ./monomer_1/protein.inp

# create protein.inp for monomer_2

cp /homes/eta/users/chong/project_covid19/common/header_a_BX_160_NX_128.inp ./protein_monomer_2.inp
cat temp_protein_monomer_2.inp >> protein_monomer_2.inp

mv protein_monomer_2.inp ./monomer_2/protein.inp

# copy solvent input file (TIP3P at 300 K)

cp /homes/eta/users/chong/project_covid19/common/tip3p_combined_300K.xsv ./BINARY/solvent.xsv
cp /homes/eta/users/chong/project_covid19/common/tip3p_combined_300K.xsv ./monomer_1/solvent.xsv
cp /homes/eta/users/chong/project_covid19/common/tip3p_combined_300K.xsv ./monomer_2/solvent.xsv

# perform 3D-RISM calculation

cd BINARY
rism3d-x protein.inp $GPU_device_num > /dev/null
cd ..

cd monomer_1
rism3d-x protein.inp $GPU_device_num > /dev/null
cd ..

cd monomer_2
rism3d-x protein.inp $GPU_device_num > /dev/null
cd ..

#
# delete unnecessary files
#

rm -rf BINARY_natom.dat
rm -rf protein.pql
rm -rf protein.pdb

rm -rf temp_protein_*.inp

rm -rf BINARY/protein.inp
rm -rf BINARY/solvent.xsv

rm -rf monomer_1/protein.inp
rm -rf monomer_1/solvent.xsv

rm -rf monomer_2/protein.inp
rm -rf monomer_2/solvent.xsv

# 
# for the next step ...
#

cd ../..

done

