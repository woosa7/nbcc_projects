
cat list_files | while read filename
do

echo $filename

cd xmua_data/$filename

#
# copy residue_info.dat
#

cp ../../../../../set_residue_info/residue_info.dat .

#
# perform xmua analysis
#

/homes/epsilon/users/nbcc/HP36_tutorial/tools/xmu_analysis/run.exe

#
# delete unnecessary files
#

rm -rf residue_info.dat

# 
# for the next step ...
#

cd ../..

done

