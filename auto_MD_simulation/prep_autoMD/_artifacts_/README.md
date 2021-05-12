# PDB Pre process 테스트

#### 1IYT
NBCC에서 테스트를 위해 사용하는 기본 단백질이며 10개의 모델을 가지고 있음

#### 1MPP
ICODE가 있음

#### 3EG3
Ligand가 있음

#### 4CK4
ALTLOC이 있음

#### 5JMU
Non standard residue가 있음

#### 5CAD
Internal missing loop 에 Non standard residue가 있음

#### USAGE
`python -m pdblib.pre_process pdblib/_artifacts_/{RCSB Code}.pdb --simple`

#### MEMO
`reduce pdblib/_artifacts_/_ligand.pdb > pdblib/_artifacts_/_ligand_h.pdb`

`antechamber -i pdblib/_artifacts_/_ligand_h.pdb -fi pdb -o pdblib/_artifacts_/_ligand.mol2 -fo mol2 -c bcc -s 2`

`parmchk -i pdblib/_artifacts_/_ligand.mol2 -f mol2 -o pdblib/_artifacts_/_ligand.mol2.frcmod`

