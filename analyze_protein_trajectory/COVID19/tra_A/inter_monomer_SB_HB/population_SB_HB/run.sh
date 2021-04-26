cp ../../set_residue_info_BINARY/BINARY_nres.dat .

dat_dir="../data/"

python population_SB_HB.py $dat_dir

rm -rf BINARY_nres.dat


# -----------------------------------------------------------------
BINARY_nres.dat

monomer_1:     1   439
monomer_2:   440   648



SB_HB_residues.dat

          21
 31 SER : 588 TYR HB
 32 ASN : 590 ALA HB
 33 TYR : 570 LEU HB



SB_ID_001_res_098_532_vs_time.dat

     30500          1
     31000          1
     31500          1



population_SB_HB.dat

  1:   98  532  SB   65.25
  2:  100  532  SB   96.00
  3:   26  592  HB   92.75
  4:   26  602  HB    5.00

# -----------------------------------------------------------------