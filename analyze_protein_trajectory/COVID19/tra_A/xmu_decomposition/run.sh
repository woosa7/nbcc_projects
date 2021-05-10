cp ../set_residue_info_BINARY/residue_info_BINARY.dat     .
cp ../set_residue_info_BINARY/residue_info_monomer_1.dat  .
cp ../set_residue_info_BINARY/residue_info_monomer_2.dat  .

dat_dir="./data/"

python run_xmua_analysis.py $dat_dir
python run_delta_xmua_analysis.py $dat_dir

rm -rf residue_info_*


"""
xmua_atm.dat

 9741
         8001.40959406   15179.78156033   -7178.37196627
    1       3.69634195       3.75987139      -0.06352944
    2      -9.66487202       0.36525319     -10.03012521
    3     -12.78864358       0.20718956     -12.99583314


xmu_res.dat

         648
    1     -49.05701279      29.02365284     -78.08066563
    2      17.63561441      21.47333770      -3.83772329
    3      17.25536374      25.61220207      -8.35683832


delta_xmu_res.dat

         648
    1      -1.47015501       0.30132179      -1.77147685
    2       0.84422837       0.66685298       0.17737540
    3       0.19556987       0.00708058       0.18848928

"""