"""
base model : bi-directional LSTM
"""

import tensorflow as tf
from geom_ops import reduce_mean_angle, dihedral_to_point, point_to_coordinate
from tensorflow.keras import layers, models, losses, optimizers, metrics, losses

NUM_DIHEDRALS = 3
NUM_DIMENSIONS = 3

max_seq_length = 700
alphabet_size = 60
lstm_cells = 800

# ==========================================================================
"""
NBCC RGN model
"""

def adam_optimizer():
    optimizer = optimizers.Adam(learning_rate=0.0001, beta_1=0.95, beta_2=0.99, epsilon=1e-07)
    return optimizer

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

    def call(self, flat_dihedrals, batch_size):
        self.num_steps = tf.cast(tf.divide(tf.shape(flat_dihedrals)[0], batch_size), dtype=tf.int32)

        # flat_dihedrals (N x batch_size, 3)
        dihedrals = tf.reshape(flat_dihedrals, [self.num_steps, batch_size, NUM_DIHEDRALS]) # (N, batch_size, 3)
        points = dihedral_to_point(dihedrals)       # (N x 3, batch_size, 3)
        coordinates = point_to_coordinate(points)   # (N x 3, batch_size, 3)
        return coordinates


class NBCC_RGN(models.Model):

    def __init__(self, batch_size):
        super(NBCC_RGN, self).__init__()

        self.batch_size = batch_size

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

    def call(self, inputs, training=True, mask=None):
        # inputs : (N, batch_size, 62)
        # -------------------------------------------------
        # RNN
        x = self.lstm1(inputs, training=training)   # (N, batch_size, 1600)
        x = self.lstm2(x, training=training)        # (N, batch_size, 1600)

        # -------------------------------------------------
        # dihedrals
        x = self.dense1(x)                          # (N, batch_size, 60)
        # print('x1 :', x.shape)

        flat_x = tf.reshape(x, [-1, 60])            # (N x batch_size, 60)
        # print('flat_x :', flat_x.shape)
        x = self.softmax(flat_x)                    # (N x batch_size, 60)
        # print('x2 :', x.shape)
        flat_dihedrals = reduce_mean_angle(x, self.alphabets) # (N x batch_size, 60) * (60, 3) = (N x batch_size, 3)
        # print('di :', flat_dihedrals.shape)

        # -------------------------------------------------
        # dihedrals to coordinates
        coordinates = self.coordinate_layer(flat_dihedrals, self.batch_size)

        # inputs :          (659, 32, 62)   :   N
        # flat_dihedrals :  (21088, 3)      :   N * 32
        # coordinates :     (1977, 32, 3)   :   N * 3

        return coordinates

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'coordinate_layer': self.coordinate_layer,
        })
        return config


# ==========================================================================
