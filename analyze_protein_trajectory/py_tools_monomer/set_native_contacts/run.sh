cp ../set_residue_info/residue_info.dat .
cp ../../initial_pdb/HP36_initial.pdb ./native_structure.pdb

python find_native_contacts.py

rm -rf residue_info.dat
