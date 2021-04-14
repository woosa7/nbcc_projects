
*** 0. installation of propka

This program requires pythyon >=3.6, so this was installed only in node105 via

  pip install propka

This can usually be used via

  propka3 input_pdb


*** 1. getting pka files with propka3

This can be done by

  sh run_propka.sh


*** 2. protonation states of HIS with propka and visual inspection by PyMOL

NOTICE: HIS residues whose pKa values are below 7.0 (assuming pH = 7.0) are
        considered neutral (i.e., either HIE or HID), whereas those whose pKa 
        values exceed 7.0 are considered protonated (i.e., HIP).

final_chain_A_kinesin_4HNA.pka: all neutral

   HIS  93 A     3.49       6.50 -> HIE (ND is H-bond acceptor)
   HIS 100 A     6.61       6.50 -> HIE (exposed to solvent)
   HIS 129 A     6.35       6.50 -> HIE (exposed to solvent)
   HIS 156 A     5.78       6.50 -> HID (exposed to solvent, but made consistent with chain D)
   HIS 191 A     5.64       6.50 -> HIE (exposed to solvent)
   HIS 200 A     6.50       6.50 -> HIE (exposed to solvent)
   HIS 205 A     3.37       6.50 -> HIE (NE is H-bond donner; ND is H-bond acceptor)

final_chain_D_kinesin_4LNU.pka: all neutral

   HIS  93 D     5.27       6.50 -> HIE (exposed to solvent)
   HIS 100 D     6.95       6.50 -> HIE (NE is H-bond donner)
   HIS 129 D     6.01       6.50 -> HIE (exposed to solvent)
   HIS 156 D     6.06       6.50 -> HID (ND is H-bond donner)
   HIS 191 D     5.51       6.50 -> HIE (exposed to solvent)
   HIS 200 D     6.46       6.50 -> HIE (exposed to solvent)
   HIS 205 D     3.69       6.50 -> HIE (NE is H-bond donner)

final_chain_B_tubulin_alpha_4HNA.pka: HIS192 protonated
(consistent with final_chain_E_tubulin_alpha_4LNU.pka)

   HIS   8 B     2.72       6.50 -> HIE (buried)
   HIS  28 B     2.12       6.50 -> HIE (buried)
   HIS  61 B     5.62       6.50 -> HIE (exposed to solvent)
   HIS  88 B     6.19       6.50 -> HIE (NE is H-bond donner in B; exposed to solvent in E)
   HIS 107 B     4.04       6.50 -> HID (ND is H-bond donner)
   HIS 139 B     3.27       6.50 -> HID (buried; to be consistent with the corresponding residue in tubulin beta)
   HIS 192 B     7.52       6.50 -> HIP (protonated according to propka; surrounded by negative charges)
   HIS 197 B     5.73       6.50 -> HID (exposed to solvent in B; ND is H-bond donner in E)
   HIS 266 B     4.10       6.50 -> HIE (NE is H-bond donner in B; buried in E)
   HIS 283 B                     -> HIE (invisible in B; ND is H-bond acceptor in E)
   HIS 309 B     6.83       6.50 -> HIE (exposed to solvent)
   HIS 393 B     6.27       6.50 -> HIE (exposed to solvent)
   HIS 406 B     6.11       6.50 -> HIE (ND is H-bond acceptor)

final_chain_C_tubulin_beta_4HNA.pka: HIS 309 protonated
(consistent with final_chain_F_tubulin_beta_4LNU.pdb)

   HIS   6 C     2.92       6.50 -> HIE (ND is H-bond acceptor)
   HIS  28 C     3.37       6.50 -> HIE (buried)
   HIS  37 C     6.51       6.50 -> HIE (exposed to solvent)
   HIS 107 C     4.19       6.50 -> HID (ND is H-bond donner)
   HIS 139 C     3.50       6.50 -> HID (buried in C; ND is H-bond donner in F)
   HIS 192 C     5.37       6.50 -> HIE (ND is H-bond acceptor in C; exposed to solvent in F)
   HIS 229 C     5.09       6.50 -> HIE (NE is H-bond donner in C; exposed to solvent in F)
   HIS 266 C     4.35       6.50 -> HIE (looks doubly protonated in C; NE is H-bond donner in F so HIE; HIE in tubulin alpha)
   HIS 309 C     7.75       6.50 -> HIP (protonated according to propka)
   HIS 406 C     6.14       6.50 -> HIE (exposed to solvent in C; ND is H-bond acceptor in F)

