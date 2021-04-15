import os
import glob
import numpy as np
from itertools import cycle
import tensorflow as tf
from utils import mask_and_weight, tertiary_loss, accumulated_loss, mapping_func, filter_func, get_data
from utils import NBCC_RGN, adam_optimizer

"""
*** output directory : outputsTest/

    id_backbone.pdb : coordinates of ProteinNet data
    id_predict.pdb  : coordinates predicted

"""

NUM_DIHEDRALS = 3

batch_size = 32

scale = 0.01   # convert to angstroms
atoms = ['N ', 'CA', 'C ']
num2aa_dict = {0: 'ALA', 1: 'CYS', 2: 'ASP', 3: 'GLU', 4: 'PHE',
               5: 'GLY', 6: 'HIS', 7: 'ILE', 8: 'LYS', 9: 'LUE',
               10: 'MET', 11: 'ASN', 12: 'PRO', 13: 'GLN', 14: 'ARG',
               15: 'SER', 16: 'THR', 17: 'VAL', 18: 'TRP', 19: 'TYR'}

output_dir = 'outputsTest'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

test_files = glob.glob('./RGN11/data/ProteinNet11/testing/*')
test_dataset  = tf.data.TFRecordDataset(tf.data.Dataset.list_files(test_files)).map(mapping_func).filter(filter_func).batch(batch_size, drop_remainder=True).prefetch(1)

# ==========================================================================
# generate RGN model and optimizer
model = NBCC_RGN(batch_size)
optimizer = adam_optimizer()

# -------------------------------------------------
# CheckpointManager
ckpt_dir = './checkpoints'
ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
ckpt_manager = tf.train.CheckpointManager(ckpt, directory=ckpt_dir, max_to_keep=5)

last_ckpt = ckpt_manager.latest_checkpoint
if last_ckpt is None:
    print('No exists checkpoint files.')
else:
    # restore weights from last checkpoint.
    print('Restored from {}'.format(last_ckpt))
    ckpt.restore(last_ckpt)

# -------------------------------------------------
# predic test dataset
_accumulated_loss_filtered = []
_accumulated_loss_factors = []

for element in test_dataset:
    features = element[0]
    ids = element[1].numpy()
    inputs, tertiaries, masks, primaries = get_data(features)
    ter_masks, weights = mask_and_weight(masks)

    # prediction
    pred_coords = model(inputs, training=False)

    # dRMSD Loss
    _, losses_filtered, loss_factors, drmsd = tertiary_loss(pred_coords, tertiaries, ter_masks, weights, batch_size)
    _accumulated_loss_filtered.extend(losses_filtered.numpy())
    _accumulated_loss_factors.extend(loss_factors.numpy())

    tertiaries = np.transpose(tertiaries, (1, 2, 0))
    pred_coords = np.transpose(pred_coords, (1, 2, 0))

    for i in range(batch_size):
        id = ids[i][0].decode('utf-8')   # byte to string

        primary = primaries[i].numpy()
        num_steps = len(primary)
        last_atom = num_steps * NUM_DIHEDRALS
        item_tertiary = tertiaries[i][:, :last_atom]
        item_prediction = pred_coords[i][:, :last_atom]

        # protein_id, # of residues, # of backbone atoms, dRMSD
        print('{}\t{}\t{}\t{:7.3f}'.format(id, num_steps, last_atom, float(drmsd[i])))
        # print(tf.squeeze(masks[i]).numpy())
        # print(primary)

        file_backbone = '{}/{}_backbone.pdb'.format(output_dir, id)
        with open(file_backbone, "w") as f:
            print('', file=f)
        file_predict = '{}/{}_predict.pdb'.format(output_dir, id)
        with open(file_predict, "w") as f:
            print('', file=f)

        # coordinates of casp data
        coord_x = item_tertiary[0,:]
        coord_y = item_tertiary[1,:]
        coord_z = item_tertiary[2,:]
        for k, (x, y, z, atom) in enumerate(zip(coord_x, coord_y, coord_z, cycle(atoms))):
            res_no = int(k/NUM_DIHEDRALS)
            aa = num2aa_dict[primary[res_no]]
            template = 'ATOM  {:>5}  {}  {} A{:>4}    {:>8.3f}{:>8.3f}{:>8.3f}  0.00  0.00           {}'.format(k+1, atom, aa, res_no+1, x*scale, y*scale, z*scale, atom[0])
            with open(file_backbone, "a") as f:
                print(template, file=f)

        # coordinates of prediction
        coord_x = item_prediction[0,:]
        coord_y = item_prediction[1,:]
        coord_z = item_prediction[2,:]
        for k, (x, y, z, atom) in enumerate(zip(coord_x, coord_y, coord_z, cycle(atoms))):
            res_no = int(k/NUM_DIHEDRALS)
            aa = num2aa_dict[primary[res_no]]
            template = 'ATOM  {:>5}  {}  {} A{:>4}    {:>8.3f}{:>8.3f}{:>8.3f}  0.00  0.00           {}'.format(k+1, atom, aa, res_no+1, x*scale, y*scale, z*scale, atom[0])
            with open(file_predict, "a") as f:
                print(template, file=f)


global_test_loss = accumulated_loss(_accumulated_loss_filtered, _accumulated_loss_factors)
print('global_test_loss :', float(global_test_loss))


"""
results of test data prediction

id, res, atoms, dRMSD
T0812	204	612	  8.665
T0824	110	330	  6.201
T0800	247	741	  8.444
T0792	80	240	  5.498
T0815	106	318	  4.772
T0782	135	405	  5.704
T0831	419	1257 19.689
T0801	376	1128  4.604
T0838	154	462	  7.950
T0790	293	879	 11.445
T0854	212	636	  4.117
T0811	255	765	  3.291
T0857	105	315	  7.914
T0821	275	825	  4.621
T0774	379	1137 12.315
T0791	300	900	 11.315
T0772	265	795	 14.109
T0835	424	1272  8.146
T0855	119	357	  6.827
T0768	170	510	  6.646
T0759	109	327	 13.386
T0829	70	210	  4.988
T0849	240	720	  4.475
T0806	258	774	  9.035
T0785	115	345	  7.567
T0767	318	954	 11.018
T0803	139	417	  8.141
T0763	163	489	  8.567
T0841	253	759	  5.268
T0837	128	384	  4.846
T0770	488	1464  7.100
T0781	420	1260 15.007
T0760	242	726	  7.813
T0845	448	1344 13.681
T0819	373	1119  5.219
T0766	130	390	  5.760
T0840	669	2007  9.942
T0776	256	768	  4.620
T0807	284	852	  4.372
T0797	44	132	  0.806
T0823	296	888	  4.104
T0851	456	1368  6.233
T0765	128	384	  7.046
T0810	383	1149 11.432
T0805	214	642	  5.844
T0817	525	1575  6.878
T0853	152	456	  9.089
T0852	414	1242 13.400
T0777	366	1098 10.602
T0764	341	1023  8.490
T0796	309	927	 15.197
T0830	575	1725 11.759
T0843	369	1107  4.869
T0832	257	771	  9.660
T0820	140	420	 16.772
T0789	295	885	 10.340
T0773	77	231	  8.210
T0786	264	792	  8.629
T0827	407	1221 15.368
T0784	155	465	  9.534
T0848	354	1062 17.393
T0771	204	612	  7.715
T0762	280	840	  5.461
T0798	198	594	  2.769

global_test_loss : 9.078693389892578
"""



# ==========================================================================
