import os
import glob
import tensorflow as tf
from utils import mask_and_weight, tertiary_loss, accumulated_loss, mapping_func, filter_func, get_data
from base_model import NBCC_RGN, adam_optimizer

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

model = NBCC_RGN(batch_size)

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

        # compute gradient and optimization
        with tf.GradientTape() as tape:
            pred_coords = model(inputs, training=True)  # LSTM

            # dRMSD Loss
            loss, losses_filtered, loss_factors, _ = tertiary_loss(pred_coords, tertiaries, ter_masks, weights, batch_size)
            loss = tf.identity(loss)

        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        # print(model.trainable_weights[-1])  # alphabets

        _accumulated_loss_filtered.extend(losses_filtered.numpy())
        _accumulated_loss_factors.extend(loss_factors.numpy())

        current_step += 1
        if current_step == 300: break

    current_epoch += 1
    global_loss = accumulated_loss(_accumulated_loss_filtered, _accumulated_loss_factors)
    # print('{:10.2f}'.format(float(global_loss)))

    with open(log_train, "a") as f:
        print('{:10.2f}'.format(float(global_loss)), file=f)

    # -------------------------------------------------
    # validation
    _accumulated_loss_filtered = []
    _accumulated_loss_factors = []

    for element in valid_dataset:
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
