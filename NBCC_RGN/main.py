import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers, metrics, losses
from geom_ops import reduce_mean_angle, dihedral_to_point, point_to_coordinate
from utils import mask_and_weight, tertiary_loss, accumulated_loss, mapping_func, filter_func, get_data

"""
python 3.7 (anaconda3-2020.02)
tensorflow 2.2

* utils.py : functions from original rgn in model.py & net_ops.py

* main_predict_testdata.py : predict coordinates from test data

"""

NUM_DIHEDRALS = 3
NUM_DIMENSIONS = 3

batch_size = 32
max_seq_length = 700
alphabet_size = 60
lstm_cells = 800

tf.random.set_seed(999)

total_epochs = 500

train_files = glob.glob('./RGN11/data/ProteinNet11/training/*')
valid_files = glob.glob('./RGN11/data/ProteinNet11/validation/*')

train_dataset = tf.data.TFRecordDataset(tf.data.Dataset.list_files(train_files, shuffle=True)).map(mapping_func).filter(filter_func).batch(batch_size, drop_remainder=True).prefetch(1)
valid_dataset = tf.data.TFRecordDataset(tf.data.Dataset.list_files(valid_files)).map(mapping_func).filter(filter_func).batch(batch_size, drop_remainder=True).prefetch(1)


# ==========================================================================
class CoordinateLayer(tf.keras.layers.Layer):
    """
    When saving the model, a RuntimeError is raised because num_steps is 'None'.
    When initializing this layer, num_steps is initialized to max_seq_length,
    then the num_steps value changed in call() for each batch.
    So when saving the model, num_steps is recognized as 'Tensor' and saved normally.
    """
    def __init__(self, num_steps):
        super(CoordinateLayer, self).__init__()
        self.num_steps = num_steps

    def call(self, flat_dihedrals):
        self.num_steps = tf.cast(tf.divide(tf.shape(flat_dihedrals)[0], batch_size), dtype=tf.int32)

        # flat_dihedrals (N x batch_size, 3)
        dihedrals = tf.reshape(flat_dihedrals, [self.num_steps, batch_size, NUM_DIHEDRALS]) # (N, batch_size, 3)
        points = dihedral_to_point(dihedrals)       # (N x 3, batch_size, 3)
        coordinates = point_to_coordinate(points)   # (N x 3, batch_size, 3)
        return coordinates


class NBCC_RGN(models.Model):

    def __init__(self):
        super(NBCC_RGN, self).__init__()

        # alphabet
        alphabet_initializer = tf.random_uniform_initializer(minval=-3.14159, maxval=3.14159)
        self.alphabets = self.add_weight(shape=(alphabet_size, NUM_DIHEDRALS), initializer=alphabet_initializer, trainable=True, name='alphabets')

        # Bi-directional LSTM
        self.lstm1 = layers.Bidirectional(layers.LSTM(lstm_cells, time_major=True, return_sequences=True, dropout=0.5), name='bi_lstm1')
        self.lstm2 = layers.Bidirectional(layers.LSTM(lstm_cells, time_major=True, return_sequences=True, dropout=0.5), name='bi_lstm2')

        # dihedrals
        self.dense1 = layers.Dense(60, name='flatten')
        self.softmax = layers.Softmax(name='softmax')

        # dihedrals to coordinates
        self.coordinate_layer = CoordinateLayer(max_seq_length) # initialize by max_seq_length

    def call(self, inputs, training=True):
        # inputs : (N, batch_size, 62)
        # -------------------------------------------------
        # RNN
        x = self.lstm1(inputs, training=training)   # (N, batch_size, 1600)
        x = self.lstm2(x, training=training)        # (N, batch_size, 1600)

        # -------------------------------------------------
        # dihedrals
        x = self.dense1(x)                          # (N, batch_size, 60)
        flat_x = tf.reshape(x, [-1, 60])            # (N x batch_size, 60)
        x = self.softmax(flat_x)                    # (N x batch_size, 60)
        flat_dihedrals = reduce_mean_angle(x, self.alphabets) # (N x batch_size, 60) * (60, 3) = (N x batch_size, 3)

        # -------------------------------------------------
        # dihedrals to coordinates
        coordinates = self.coordinate_layer(flat_dihedrals)

        return coordinates

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'coordinate_layer': self.coordinate_layer,
        })
        return config


# ==========================================================================
model = NBCC_RGN()
optimizer = optimizers.Adam(learning_rate=0.0001, beta_1=0.95, beta_2=0.99, epsilon=1e-07)

log_file = 'loss_train_valid'

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
            pred_coords = model(inputs)

            # dRMSD Loss
            loss, losses_filtered, loss_factors, _ = tertiary_loss(pred_coords, tertiaries, ter_masks, weights)
            loss = tf.identity(loss)

        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        # print(model.trainable_weights[-1])  # alphabets

        _accumulated_loss_filtered.extend(losses_filtered.numpy())
        _accumulated_loss_factors.extend(loss_factors.numpy())

        current_step += 1
        # if current_step == 3: break

    current_epoch += 1
    global_loss = accumulated_loss(_accumulated_loss_filtered, _accumulated_loss_factors)

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
        valid_loss, losses_filtered, loss_factors, _ = tertiary_loss(pred_coords, tertiaries, ter_masks, weights)

        _accumulated_loss_filtered.extend(losses_filtered.numpy())
        _accumulated_loss_factors.extend(loss_factors.numpy())

    global_valid_loss = accumulated_loss(_accumulated_loss_filtered, _accumulated_loss_factors)

    # -------------------------------------------------
    # epoch, train_loss, valid_loss
    log_str = '{:03d},\t{:10.2f},\t{:10.2f}'.format(current_epoch, float(global_loss), float(global_valid_loss))
    print(log_str)
    with open(log_file, "a") as f:
        print(log_str, file=f)

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

    # -------------------------------------------------

# ==========================================================================
