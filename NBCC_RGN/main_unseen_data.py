import os
import glob
import subprocess
import numpy as np
from itertools import cycle
import tensorflow as tf
from convert_to_tfrecord import read_record, dict_to_tfrecord
from utils import mask_and_weight, tertiary_loss, accumulated_loss, mapping_func, filter_func, get_data
from utils import NBCC_RGN, adam_optimizer

"""
Predict the tertiary structure of a unseen protein.

*** Generate input data from unseen protein sequence, and then predict it's coordinates

(1) Install HMMER

    $apt install hmmer                              (need root account)
    or
    $conda install -c bioconda hmmer                (in virtual environment)

(2) Download the raw sequence databases. (zip 23 GB. unzip 51 GB.)
    Download in "data_processing" directory.

    $wget https://sharehost.hms.harvard.edu/sysbio/alquraishi/proteinnet/sequence_dbs/proteinnet11.gz
    $gunzip proteinnet11.gz

(3) copy fasta file of unseen protein to 'data_processing' directory

(4) run python program

    $python main_unseen_data.py

"""

# ==========================================================================
# convert functions from original RGN

def convert_to_proteinnet(stem):
    """
    this code from original rgn.
    modified by NBCC
    """
    i_fa = open(stem + '.fasta', 'r')
    name = i_fa.readline()[1:]
    seq  = "".join([line.strip() for line in i_fa.readlines()]) + '\n'
    header = '[ID]\n' + name + '[PRIMARY]\n' + seq + '[EVOLUTIONARY]'
    i_fa.close()

    i_icinfo = open(stem + '.icinfo', 'r')
    i_cinfo  = open(stem + '.cinfo', 'r')
    evos = []
    for buf_icinfo in range(9): buf_icinfo = i_icinfo.readline()
    for buf_cinfo in range(10): buf_cinfo  = i_cinfo.readline()

    while buf_icinfo != '//\n':
    	buf_icinfo_split = buf_icinfo.split()
    	if buf_icinfo_split[0] != '-':
    		ps = np.array([float(p) for p in buf_cinfo.split()[1:]])
    		ps = ps / np.sum(ps)
    		evo = np.append(ps, float(buf_icinfo_split[3]) / np.log2(20))
    		evos.append(np.tile(evo, 2))
    	buf_icinfo = i_icinfo.readline()
    	buf_cinfo  = i_cinfo.readline()

    i_icinfo.close()
    i_cinfo.close()

    np.savetxt(stem + '.proteinnet', np.stack(evos).T, fmt='%1.5f', comments='', header=header)

    mask = '+' * (len(seq)-1) + '\n'
    seq_mask = '[MASK]\n' + mask

    with open(stem + '.proteinnet', 'a') as o:
        o.write(seq_mask)
        o.write('\n')

def convert_to_tfrecord(stem):
    """
    this code from original rgn.
    modified by NBCC
    """
    input_path      = stem + '.proteinnet'
    output_path     = stem + '.tfrecord'
    num_evo_entries = 42     # default number of evo entries

    input_file = open(input_path, 'r')
    output_file = tf.io.TFRecordWriter(output_path)

    dict_ = read_record(input_file, num_evo_entries)

    if dict_ is not None:
        tfrecord_serialized = dict_to_tfrecord(dict_).SerializeToString()
        output_file.write(tfrecord_serialized)
    else:
        input_file.close()
        output_file.close()

# ==========================================================================
"""
* Convert unseen protein fasta file to proteinnet file and save it in tfrecord format.
"""

work_dir = 'data_processing'
output_dir = 'outputsTest'

root_dir = os.getcwd()
os.chdir(work_dir)

Z_VALUE=1e8
E_VALUE=1e-10

fasta_files = glob.glob('*.fasta')
for file in fasta_files:
    stem = os.path.basename(file).split('.')[0]

    # (3) transform a unseen protein sequence to RGN input data format.
    cmd_jack = 'jackhmmer -N 5 -Z {0} --incE {1} --incdomE {1} -E {1} --domE {1} --cpu 8 -o /dev/null -A {2}.sto --tblout {2}.tblout {2}.fasta proteinnet11'.format(Z_VALUE, E_VALUE, stem)
    cmd_reformat = 'esl-reformat -o {0}.a2m a2m {0}.sto'.format(stem)
    cmd_weight   = 'esl-weight -p --amino --informat a2m -o {0}.weighted.sto {0}.a2m'.format(stem)
    cmd_alistat  = 'esl-alistat --weight --amino --icinfo {0}.icinfo --cinfo {0}.cinfo {0}.weighted.sto > /dev/null'.format(stem)

    if not os.path.exists('%s.tblout' % stem):
        # elapsed time : about 2 hours for Ab42
        proc = subprocess.Popen(cmd_jack, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
        print(proc.stdout.read())

        proc = subprocess.call(cmd_reformat, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
        if proc != 0: print(proc)
        proc = subprocess.call(cmd_weight, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
        if proc != 0: print(proc)
        proc = subprocess.call(cmd_alistat, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
        if proc != 0: print(proc)

    if not os.path.exists('%s.tfrecord' % stem):
        convert_to_proteinnet(stem)
        convert_to_tfrecord(stem)


print(root_dir)
os.chdir(root_dir)

# copy tfrecord files to output directory
subprocess.call('cp {}/*.tfrecord {}/'.format(work_dir, output_dir), shell=True)


# ==========================================================================
"""
* Predict the tertiary structure of unseen protein using the saved model.
"""

NUM_DIHEDRALS = 3

batch_size = 1

scale = 0.01   # convert to angstroms
atoms = ['N ', 'CA', 'C ']
num2aa_dict = {0: 'ALA', 1: 'CYS', 2: 'ASP', 3: 'GLU', 4: 'PHE',
               5: 'GLY', 6: 'HIS', 7: 'ILE', 8: 'LYS', 9: 'LUE',
               10: 'MET', 11: 'ASN', 12: 'PRO', 13: 'GLN', 14: 'ARG',
               15: 'SER', 16: 'THR', 17: 'VAL', 18: 'TRP', 19: 'TYR'}


test_files = glob.glob('{}/*.tfrecord'.format(output_dir))
test_dataset  = tf.data.TFRecordDataset(tf.data.Dataset.list_files(test_files)).map(mapping_func).batch(batch_size)

# ==========================================================================
# generate RGN model and optimizer
model = NBCC_RGN(batch_size)
optimizer = adam_optimizer()

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
# predic coordinates
for element in test_dataset:
    id = element[1].numpy()
    id = id[0][0].decode('utf-8')   # byte to string

    features = element[0]
    inputs, tertiaries, masks, primaries = get_data(features)
    ter_masks, weights = mask_and_weight(masks)

    # prediction
    pred_coords = model(inputs, training=False)
    pred_coords = np.transpose(pred_coords, (1, 2, 0))

    primary = primaries[0].numpy()
    num_steps = len(primary)
    last_atom = num_steps * NUM_DIHEDRALS
    item_prediction = pred_coords[0][:, :last_atom]

    # protein_id, # of residues, # of backbone atoms, dRMSD
    print('{}\t{}\t{}'.format(id, num_steps, last_atom))

    file_predict = '{}/{}_predict.pdb'.format(output_dir, id)
    with open(file_predict, "w") as f:
        print('', file=f)

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

# ==========================================================================
