cp ../set_residue_info/residue_info.dat .
cp ../set_native_contacts/native_contacts_atoms.dat .

dat_dir="../../traj_A/pdb_from_crd/"

python run_Q_vs_time.py $dat_dir

# rm -rf residue_info.dat
# rm -rf native_contacts_atoms.dat
