cp ../set_residue_info_BINARY/residue_info_BINARY.dat     .
cp ../set_residue_info_BINARY/residue_info_monomer_1.dat  .
cp ../set_residue_info_BINARY/residue_info_monomer_2.dat  .

dat_dir="./data/"

python run_xmua_analysis.py $dat_dir
python run_delta_xmua_analysis.py $dat_dir

rm -rf residue_info_*
