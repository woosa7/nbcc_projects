import os
import glob
import tensorflow as tf
from utils import mask_and_weight, tertiary_loss, accumulated_loss, mapping_func, filter_func, get_data
from attention_model import NBCC_RGN_Attention, adam_optimizer, NBCC_RGN_Attention_3

"""
python 3.7 (anaconda3-2020.02)
tensorflow 2.2

* utils.py : model and functions

* main_predict_testdata.py : predict coordinates from test data

* main_unseen_data.py : predict a tertiary structure of unseen protein

* convert_to_tfrecord.py : convert fasta to tfrecord. copy from original rgn and modified by NBCC

"""

tf.random.set_seed(999)

batch_size = 32
total_epochs = 500

train_files = glob.glob('./RGN11/data/ProteinNet11/training/*')
valid_files = glob.glob('./RGN11/data/ProteinNet11/validation/*')
train_dataset = tf.data.TFRecordDataset(tf.data.Dataset.list_files(train_files, shuffle=True)).map(mapping_func).filter(filter_func).batch(batch_size, drop_remainder=True).prefetch(1)
valid_dataset = tf.data.TFRecordDataset(tf.data.Dataset.list_files(valid_files)).map(mapping_func).filter(filter_func).batch(batch_size, drop_remainder=True).prefetch(1)


# ==========================================================================
# generate RGN model and optimizer
optimizer = adam_optimizer()

# Attention Network
model = NBCC_RGN_Attention(batch_size)
# model = NBCC_RGN_Attention_3(batch_size)

# -------------------------------------------------
# CheckpointManager
ckpt_dir = './checkpoints'
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
ckpt_manager = tf.train.CheckpointManager(ckpt, directory=ckpt_dir, max_to_keep=5)

last_ckpt = ckpt_manager.latest_checkpoint
if last_ckpt is None:
    current_epoch = 0
else:
    # restore weights from last checkpoint.
    print('Restored from {}'.format(last_ckpt))
    ckpt.restore(last_ckpt)
    current_epoch = int(last_ckpt.split('-')[1])
    # min_valid_loss
    with open('{}/ckpt-{}.loss'.format(ckpt_dir, current_epoch), "r") as f:
        min_valid_loss = float(f.readlines()[0])

# -------------------------------------------------
# training and validation steps
log_train = 'loss_train'
log_valid = 'loss_valid'

for epoch in range(current_epoch, total_epochs, 1):
    # -------------------------------------------------
    # training
    current_step = 0
    _accumulated_loss_filtered = []
    _accumulated_loss_factors = []

    for element in train_dataset:
        features = element[0]
        ids = element[1]
        inputs, tertiaries, masks, _ = get_data(features)
        ter_masks, weights = mask_and_weight(masks)

        inputs = tf.transpose(inputs, perm=(1, 0, 2))
        # print('inputs :', inputs.shape)                # (32,  659, 62)
        # print('tertiaries :', tertiaries.shape)        # (32, 1977, 3)

        # ----------------------------------------------------------------------
        # for k in range(inputs.shape[0]):
        #     prot = inputs[k,:,:]
        #
        #     for j in range(prot.shape[0]):
        #         residue = prot[j,:]
        #         # atom_type
        #         atom1 = tf.concat([residue, tf.constant([1.0,0.0,0.0])], axis=0)    # N
        #         atom2 = tf.concat([residue, tf.constant([0.0,1.0,0.0])], axis=0)    # CA
        #         atom3 = tf.concat([residue, tf.constant([0.0,0.0,1.0])], axis=0)    # C
        #         atom1 = tf.expand_dims(atom1, axis=0)
        #         atom2 = tf.expand_dims(atom2, axis=0)
        #         atom3 = tf.expand_dims(atom3, axis=0)
        #
        #         if j == 0:
        #             atoms = tf.concat([atom1, atom2, atom3], axis=0)
        #         else:
        #             atoms = tf.concat([atoms, atom1, atom2, atom3], axis=0)
        #
        #     atoms = tf.expand_dims(atoms, axis=0)
        #
        #     if k == 0:
        #         inputs2 = tf.identity(atoms)
        #     else:
        #         inputs2 = tf.concat([inputs2, atoms], axis=0)

        # inputs2 : (32, 1977, 65)
        # print('inputs2 :', inputs2.shape)
        # ----------------------------------------------------------------------

        # compute gradient and optimization
        with tf.GradientTape() as tape:
            sos = tf.ones(shape=[1, 32, 3], dtype=tf.float32)
            labels = tf.concat([sos, tertiaries], axis=0)
            shifted_labels = labels[:-1,:,:]
            shifted_labels = tf.transpose(shifted_labels, perm=(1, 0, 2))     # (batch, N, 3)
            pred_coords = model([inputs, shifted_labels], training=True)

            # pred_coords = model([inputs2, tertiaries], training=True)

            # dRMSD Loss
            loss, losses_filtered, loss_factors, _ = tertiary_loss(pred_coords, tertiaries, ter_masks, weights, batch_size)
            loss = tf.identity(loss)

        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        # print(model.trainable_weights[-1])  # alphabets

        _accumulated_loss_filtered.extend(losses_filtered.numpy())
        _accumulated_loss_factors.extend(loss_factors.numpy())

        current_step += 1
        if current_step == 30: break


    current_epoch += 1
    global_loss = accumulated_loss(_accumulated_loss_filtered, _accumulated_loss_factors)

    # print('global_loss: {:10.2f}'.format(float(global_loss)))

    with open(log_train, "a") as f:
        print('{:10.2f}'.format(float(global_loss)), file=f)

    # -------------------------------------------------
    # validation
    _accumulated_loss_filtered = []
    _accumulated_loss_factors = []

    for element in valid_dataset.take(1):
        features = element[0]
        ids = element[1]
        inputs, tertiaries, masks, _ = get_data(features)
        ter_masks, weights = mask_and_weight(masks)
    
        # prediction
        pred_coords = model(inputs, training=False)


        # dRMSD Loss
        valid_loss, losses_filtered, loss_factors, _ = tertiary_loss(pred_coords, tertiaries, ter_masks, weights, batch_size)

        _accumulated_loss_filtered.extend(losses_filtered.numpy())
        _accumulated_loss_factors.extend(loss_factors.numpy())

    global_valid_loss = accumulated_loss(_accumulated_loss_filtered, _accumulated_loss_factors)

    # -------------------------------------------------
    # epoch, train_loss, valid_loss
    log_str = '{:03d},\t{:10.2f},\t{:10.2f}'.format(current_epoch, float(global_loss), float(global_valid_loss))
    print(log_str)

    with open(log_valid, "a") as f:
        print('{:10.2f}'.format(float(global_valid_loss)), file=f)

    # -------------------------------------------------
    # save checkpoint
    if current_epoch == 1:
        min_valid_loss = global_valid_loss
    else:
        if min_valid_loss > global_valid_loss:
            min_valid_loss = global_valid_loss
            with open('{}/ckpt-{}.loss'.format(ckpt_dir, current_epoch), "w") as f:
                print(float(min_valid_loss), file=f)

            ckpt_manager.save(checkpoint_number=current_epoch)

# ==========================================================================
