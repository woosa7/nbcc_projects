export AMBERHOME=/opt/apps/amber20/amber20
export CUDA_HOME=/opt/cuda-10.0

export LD_LIBRARY_PATH=/usr/local/lib:$CUDA_HOME/lib64/:$CUDA_HOME/lib:$AMBERHOME/lib
export PATH=/usr/local/bin:/usr/bin:/bin:$AMBERHOME/bin:$CUDA_HOME/bin/:/opt/nbcc/bin:$HOME/bin

#
# set GPU device number (either 0 or 1)
#

GPU_device_num=0

#
# do xmu-decom calculation (type a) for each pdb file in list_files
#

cat list_files | while read filename
do

echo $filename

cd xmua_data

mkdir $filename
cd $filename

#
# copy protein.pql (parameter only) and protein.pdb
#

cp ../../../../../set_pql_param/protein.pql .
cp ../../../../pdb_from_crd/"$filename".pdb ./protein.pdb

#
# create temp_protein.inp
#
#   input : protein.pql (parameter only)
#           protein.pdb
#
#   output: temp_protein.inp
#

../../../../../tools/mkrsm_from_pql_and_pdb/mkrsm_monomer/run.exe

#
# create protein.inp
#

cp /opt/nbcc/common/3D-RISM/header_a_BX_128_NX_128.inp ./protein.inp
cat temp_protein.inp >> protein.inp

#
# copy solvent input file
#

cp /opt/nbcc/common/3D-RISM/tip3p_combined_300K.xsv ./solvent.xsv

#
# perform 3D-RISM calculation
#

rism3d-x protein.inp $GPU_device_num > /dev/null

#
# delete unnecessary files
#

rm -rf protein.pql
rm -rf protein.pdb

rm -rf temp_protein.inp
rm -rf solvent.xsv

rm -rf protein.inp

# 
# for the next step ...
#

cd ../..

done

