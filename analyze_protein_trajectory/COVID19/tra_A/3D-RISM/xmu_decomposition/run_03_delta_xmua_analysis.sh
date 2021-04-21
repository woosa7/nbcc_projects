
cat list_files | while read filename
do

echo $filename

cd data/$filename

#
# make delta directory if it does not exist
#

if [ ! -d delta ]; then
  mkdir delta
fi

#
# copy xmua_atm.dat files
#

cp ./BINARY/xmua_atm.dat    ./delta/xmua_atm_BINARY.dat
cp ./monomer_1/xmua_atm.dat ./delta/xmua_atm_monomer_1.dat
cp ./monomer_2/xmua_atm.dat ./delta/xmua_atm_monomer_2.dat

#
# copy residue_info.dat files
#

cp /homes/eta/users/chong/project_covid19/CV30_RBD/set_residue_info_BINARY/residue_info_BINARY.dat    ./delta/
cp /homes/eta/users/chong/project_covid19/CV30_RBD/set_residue_info_BINARY/residue_info_monomer_1.dat ./delta/
cp /homes/eta/users/chong/project_covid19/CV30_RBD/set_residue_info_BINARY/residue_info_monomer_2.dat ./delta/

#
# perform HI_thma analysis
#

cd delta

  /homes/eta/users/chong/project_covid19/tools/delta_xmu_analysis/run.exe

cd ..

#
# delete unnecessary files
#

rm -rf ./delta/*_BINARY.dat
rm -rf ./delta/*_monomer_1.dat
rm -rf ./delta/*_monomer_2.dat

# 
# for the next step ...
#

cd ../..

done

