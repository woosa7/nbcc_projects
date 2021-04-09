
cat list_files | while read filename
do

echo $filename

cd thma_data/$filename

#
# copy residue_info.dat
#

cp ../../../../../set_residue_info/residue_info.dat .

#
# perform thma analysis
#

/homes/epsilon/users/nbcc/HP36_tutorial/tools/xmu_analysis/run.exe
/homes/epsilon/users/nbcc/HP36_tutorial/tools/ene_analysis/run.exe
/homes/epsilon/users/nbcc/HP36_tutorial/tools/ent_analysis/run.exe
/homes/epsilon/users/nbcc/HP36_tutorial/tools/euv_analysis/run.exe
/homes/epsilon/users/nbcc/HP36_tutorial/tools/evv_analysis/run.exe

#
# delete unnecessary files
#

rm -rf residue_info.dat

# 
# for the next step ...
#

cd ../..

done

