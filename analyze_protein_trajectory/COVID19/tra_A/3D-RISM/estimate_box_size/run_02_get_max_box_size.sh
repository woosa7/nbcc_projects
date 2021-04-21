
echo "the largest 10 xmax" > temp_x.dat
sort -n -r -k3 out.log | head -10 >> temp_x.dat

echo "the largest 10 ymax" >> temp_y.dat
sort -n -r -k6 out.log | head -10 >> temp_y.dat

echo "the largest 10 zmax" > temp_z.dat
sort -n -r -k9 out.log | head -10 >> temp_z.dat

cat temp_x.dat >  max_box_size.dat
cat temp_y.dat >> max_box_size.dat
cat temp_z.dat >> max_box_size.dat

rm -rf temp_x.dat
rm -rf temp_y.dat
rm -rf temp_z.dat


