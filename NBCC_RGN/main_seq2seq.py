import os
import sys
import glob
import tensorflow as tf
from utils import mask_and_weight, tertiary_loss, accumulated_loss, mapping_func, filter_func, get_data
from attention_model import NBCC_RGN_seq2seq, adam_optimizer
from tensorflow.keras import preprocessing


"""
python 3.7 (anaconda3-2020.02)
tensorflow 2.2
"""

tf.get_logger().setLevel('ERROR')

tf.random.set_seed(999)

batch_size = 32
total_epochs = 500

test_batch_size = 1

max_seq_length = 700

train_files = glob.glob('./RGN11/data/ProteinNet11/training/*')
valid_files = glob.glob('./RGN11/data/ProteinNet11/validation/*')
train_dataset = tf.data.TFRecordDataset(tf.data.Dataset.list_files(train_files, shuffle=True)).map(mapping_func).filter(filter_func).batch(batch_size, drop_remainder=True).prefetch(1)
valid_dataset = tf.data.TFRecordDataset(tf.data.Dataset.list_files(valid_files)).map(mapping_func).filter(filter_func).batch(test_batch_size, drop_remainder=True).prefetch(1)


# ==========================================================================
# generate RGN model and optimizer
optimizer = adam_optimizer()

model = NBCC_RGN_seq2seq(batch_size)

# -------------------------------------------------
# training and validation steps
log_train = 'loss_train'
log_valid = 'loss_valid'

current_epoch = 0

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

        # ----------------------------------------------------------------------
        inputs = tf.transpose(inputs, perm=(1, 0, 2))
        tertiaries = tf.transpose(tertiaries, perm=(1, 0, 2))
        # print('inputs:', inputs.shape) # (32, 659, 62)

        labels = tf.reshape(tertiaries, shape=[batch_size, inputs.shape[1], 3, 3])
        labels = tf.reshape(labels, shape=[labels.shape[0], labels.shape[1], -1])

        sos = tf.ones(shape=[batch_size, 1, 9], dtype=tf.float32)
        labels = tf.concat([sos, labels], axis=1)
        # print('inputs:', inputs.shape) # (32, 659, 62)
        # print('labels:', labels.shape) # (32, 660, 9)
        # a = labels[0, 0, :]
        # print(a)

        output_labels  = labels[:, 1:, :]
        shifted_labels = labels[:, :-1, :]  # 학습시 입력으로 사용

        # compute gradient and optimization
        with tf.GradientTape() as tape:
            pred_coords = model([inputs, shifted_labels], training=True)
            # print('pred_coords :', pred_coords.shape)
            # a = pred_coords[0, 0, :]
            # print(a)

            # (batch, N, 9) --> (batch, N * 3, 3)
            # pred_coords = tf.reshape(pred_coords, shape=[batch_size, pred_coords.shape[1], 3, 3])
            # pred_coords = tf.reshape(pred_coords, shape=[batch_size, -1, 3])
            # pred_coords = tf.transpose(pred_coords, perm=(1, 0, 2))

            tertiaries = tf.transpose(tertiaries, perm=(1, 0, 2))
            # print('tertiaries  :', tertiaries.shape)
            # print('pred_coords :', pred_coords.shape)

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
    global_loss = accumulated_loss(_accumulated_loss_filtered, _accumulated_loss_factors) * 10000
    print('{} - global_loss: {:10.2f}'.format(current_epoch, float(global_loss)))

    with open(log_train, "a") as f:
        print('{:10.2f}'.format(float(global_loss)), file=f)

    # -------------------------------------------------
    # validation
    _accumulated_loss_filtered = []
    _accumulated_loss_factors = []

    for element in valid_dataset.take(3):
        features = element[0]
        ids = element[1]
        inputs, tertiaries, masks, _ = get_data(features)
        ter_masks, weights = mask_and_weight(masks)

        # ----------------------------------------------------------------------
        inputs = tf.transpose(inputs, perm=(1, 0, 2))
        # tertiaries = tf.transpose(tertiaries, perm=(1, 0, 2))

        # ----------------------------------------------------------------------
        # prediction
        pred_coords = model(inputs, training=False)
        # a = pred_coords[0, 0, :]
        # print(a)

        pred_coords = tf.reshape(pred_coords, shape=[1, pred_coords.shape[1], 3, 3])
        pred_coords = tf.reshape(pred_coords, shape=[1, -1, 3])
        pred_coords = tf.transpose(pred_coords, perm=(1, 0, 2))
        # print('pred_coords :', pred_coords.shape)

        # dRMSD Loss
        valid_loss, losses_filtered, loss_factors, _ = tertiary_loss(pred_coords, tertiaries, ter_masks, weights, 1)

        _accumulated_loss_filtered.extend(losses_filtered.numpy())
        _accumulated_loss_factors.extend(loss_factors.numpy())

    global_valid_loss = accumulated_loss(_accumulated_loss_filtered, _accumulated_loss_factors) * 10000

    # -------------------------------------------------
    # epoch, train_loss, valid_loss
    log_str = '{:03d},\t{:10.2f},\t{:10.2f}'.format(current_epoch, float(global_loss), float(global_valid_loss))
    print(log_str)

    with open(log_valid, "a") as f:
        print('{:10.2f}'.format(float(global_valid_loss)), file=f)


# ==========================================================================
