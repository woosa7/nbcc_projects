cp ../set_residue_info_BINARY/BINARY_natom.dat .
cp ../set_residue_info_BINARY/residue_info_BINARY.dat ./residue_info.dat

dat_dir="../../CV30_RBD/traj_A/pdb_from_crd/"

python find_inter_monomer_heavy_atom_contacts.py $dat_dir

rm -rf BINARY_natom.dat
rm -rf residue_info.dat
