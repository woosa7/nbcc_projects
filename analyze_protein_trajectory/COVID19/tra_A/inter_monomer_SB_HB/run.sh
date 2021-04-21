# main part

cp ../set_residue_info_BINARY/BINARY_natom.dat .
cp ../set_residue_info_BINARY/residue_info_BINARY.dat ./residue_info.dat

dat_dir="../../CV30_RBD/traj_A/pdb_from_crd/"

python find_inter_monomer_SB_HB.py $dat_dir

rm -rf BINARY_natom.dat
rm -rf residue_info.dat


#---------------------------------
residue_info

 648
   1     1    17
   2    18    33
   3    34    50
   4    51    69
   5    70    85


SB_HB_atoms.dat

          24
  420  O    31 SER :  8860  OH  588 TYR :    2.666 HB
  430  ND2  32 ASN :  8894  O   590 ALA :    2.770 HB
  448  OH   33 TYR :  8551  O   570 LEU :    2.712 HB


SB_HB_residues.dat

          21
 31 SER : 588 TYR HB
 32 ASN : 590 ALA HB
 33 TYR : 570 LEU HB
