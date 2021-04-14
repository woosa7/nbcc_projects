> tleap commands

         source /opt/apps/amber20/amber20/dat/leap/cmd/leaprc.protein.ff14SB
         source /opt/apps/amber20/amber20/dat/leap/cmd/leaprc.water.tip3p
         prot = loadpdb ../00_pdb_files/cv30com_final.pdb
         desc prot
	 bond prot.22.SG   prot.95.SG
         bond prot.145.SG  prot.201.SG
         bond prot.247.SG  prot.313.SG
         bond prot.359.SG  prot.419.SG
         bond prot.451.SG  prot.476.SG
         bond prot.494.SG  prot.547.SG
	 bond prot.506.SG  prot.640.SG
	 bond prot.595.SG  prot.603.SG
         solvateBox prot TIP3PBOX 15.0 
         addIons prot Na+ 0
         addIons prot Cl- 0
         saveAmberParm prot cv30com_rbd.top cv30com_rbd_initial.crd
         quit



> making pdb


        ambpdb -p cv30com_rbd.top < cv30com_rbd_initial.crd > cv30com_rbd_initial.pdb




> check topology file


        2.03800000E+00

        must be present in %FLAG BOND_EQUIL_VALUE.


