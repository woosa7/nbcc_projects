cp ../../set_residue_info_BINARY/residue_info_BINARY.dat     .
cp ../../set_residue_info_BINARY/residue_info_monomer_1.dat  .
cp ../../set_residue_info_BINARY/residue_info_monomer_2.dat  .

dat_dir="../data/"

python analytic_xmu_vs_time.py $dat_dir
python xmu_vs_time.py $dat_dir

rm -rf residue_info_*
