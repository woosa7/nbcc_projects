"""
Attention network model (with LSTM)

main_attention.py

attention_model.py
* NBCC_RGN_Attention_3 : Seq-to-Seq
* NBCC_RGN_Attention_2 : Seq-to-Seq with Attention. predict the coordinates
* NBCC_RGN_Attention   : Seq-to-Seq with Attention. predict the positions and alphabets and then translate to coordinates.

All three models show similar patterns in train loss and validation loss.

I'm going to try different structured prediction models.

"""

import sys
import tensorflow as tf
from geom_ops import reduce_mean_angle, dihedral_to_point, point_to_coordinate
from tensorflow.keras import layers, models, losses, optimizers, metrics, losses

NUM_DIHEDRALS = 3
NUM_DIMENSIONS = 3

max_seq_length = 700
alphabet_size = 60
lstm_cells = 800

# ==========================================================================
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


class Encoder(tf.keras.Model):
    def __init__(self):
        super(Encoder, self).__init__()

        self.lstm_forward  = layers.LSTM(lstm_cells, return_sequences=True, return_state=True, dropout=0.5)
        self.lstm_backward = layers.LSTM(lstm_cells, return_sequences=True, return_state=True, dropout=0.5, go_backwards=True)

    def call(self, x, training=True):
        # output, hidden state, cell state
        H1, h1, c1 = self.lstm_forward(x)
        H2, h2, c2 = self.lstm_backward(x)
        H = tf.concat([H1, H2], axis=2)     # (batch, N, 1600)
        h = tf.concat([h1, h2], axis=1)     # (batch, 1600)
        c = tf.concat([c1, c2], axis=1)     # (batch, 1600)

        # H, h, c = self.lstm_forward(x)

        return H, h, c

class Decoder(tf.keras.Model):
    def __init__(self, batch_size):
        super(Decoder, self).__init__()

        self.batch_size = batch_size

        self.lstm = layers.LSTM(lstm_cells * 2, return_sequences=True, return_state=True, dropout=0.5)

        # alphabet
        alphabet_initializer = tf.random_uniform_initializer(minval=-3.14159, maxval=3.14159)
        self.alphabets = self.add_weight(shape=(alphabet_size, NUM_DIHEDRALS), initializer=alphabet_initializer, trainable=True, name='alphabets')

        # Attention layer !!!
        self.attention = layers.Attention()

        self.dense1 = layers.Dense(60, name='flatten')
        self.softmax = layers.Softmax(name='softmax')

        # dihedrals to coordinates
        self.coordinate_layer = CoordinateLayer(max_seq_length) # initialize by max_seq_length

    def call(self, inputs, training=True):
        # y  : ground truth                 (batch, N*3 , 3)
        # s0 : hidden state of Encoder      (batch, 1600)
        # c0 : cell state of Encoder        (batch, 1600)
        # H  : output of Encoder            (batch, N, 1600)
        y, s0, c0, H_value = inputs

        # ------------------------------------
        S, h, c = self.lstm(y, initial_state=[s0, c0])  # S : all hidden states for y. (batch, N, 1600)

        # ------------------------------------
        # H_value : output of Encoder --> Value of Attention
        S_key = tf.concat([s0[:, tf.newaxis, :], S[:, :-1, :]], axis=1) # --> Key of Attention
        A = self.attention([S_key, H_value])    # key, value
        y_ = tf.concat([S, A], axis=-1) #  (32, N*3, 3200)
        x = self.dense1(y_)                          # (32, N, 60)

        # ------------------------------------
        # x = self.dense1(S)                          # (32, N, 60)
        # print('x1 :', x.shape)  # 1977 --> 659

        flat_x = tf.reshape(x, [-1, 60])            # (N x batch_size, 60)
        # print('flat_x :', flat_x.shape) # 21088, 60

        x = self.softmax(flat_x)                    # (N x batch_size, 60)
        # print('x2 :', x.shape) # 21088, 60

        flat_dihedrals = reduce_mean_angle(x, self.alphabets) # (N x batch_size, 60) * (60, 3) = (N x batch_size, 3)
        # print('di :', flat_dihedrals.shape) # 21088, 3

        # dihedrals to coordinates
        if training:
            batch_size = self.batch_size
        else:
            batch_size = 1

        coordinates = self.coordinate_layer(flat_dihedrals, batch_size)
        # print('coordinates:', coordinates.shape)
        # print('coordinates:', coordinates[0:3, 0, :])


        return coordinates, h, c


class NBCC_RGN_Attention(models.Model):

    def __init__(self, batch_size):
        super(NBCC_RGN_Attention, self).__init__()

        self.batch_size = batch_size

        # Attention Network
        self.enc = Encoder()
        self.dec = Decoder(self.batch_size)
        self.sos = 1.0

    def call(self, inputs, training=False, mask=None):

        # N = residues * 3
        if training is True:
            x, y = inputs
            # x = tf.transpose(x, perm=(1, 0, 2))     # (batch, N, 62)
            # y = tf.transpose(y, perm=(1, 0, 2))     # (batch, N*3, 3)
            # print('x:', x.shape)
            # print('y:', y.shape)

            """
            Gradient vanishing of RNN is solved by collecting all hidden states of the encoder
            and passing them to the decoder.

            Query and key-value pair.
            Compare which K is similar to Q and synthesize V based on the similarity.

            Encoder's hidden layers are used as key and value.
            The hidden layers of the decoder (one of the previous timesteps) are used as query.
            """

            H, h, c = self.enc(x)   # output, hidden state, cell state

            coordinates, _, _ = self.dec((y, h, c, H))

            return coordinates

        else:
            x = inputs
            coord_len = x.shape[1]

            H, h, c = self.enc(x)

            # -------------------------------------------------------------------------
            # ver. 1
            test_batch_size = 1
            y = tf.ones(shape=[test_batch_size, 1, 9], dtype=tf.float32) # one residue : N-CA-C * 3
            # y = tf.random.uniform(shape=[test_batch_size, 1, 1], dtype=tf.float32) # one residue : N-CA-C * 3

            # print('valid - x:', x.shape)
            # print('valid - y:', y.shape)

            for idx in tf.range(coord_len):
                y_, h, c = self.dec((y, h, c, H)) # (3, 1, 3)

                # print('y_', y_)
                # print(y_[0, :, :])

                # y를 계속 다음 입력으로 사용
                # y : (3, 1, 3) --> (1, 3, 3) --> (1, 1, 9)
                y = tf.transpose(y_, perm=(1, 0, 2))
                y = tf.reshape(y, shape=[test_batch_size, 1, 3, 3])
                y = tf.reshape(y, shape=[1, 1, -1])

                if idx == 0:
                    coordinates = tf.identity(y_)
                else:
                    coordinates = tf.concat([coordinates, y_], axis=0)

            # -------------------------------------------------------------------------
            # ver. 2
            # y = tf.ones(shape=[1, coord_len, 9], dtype=tf.float32)
            # coordinates, _, _ = self.dec((y, h, c, H))

            # -------------------------------------------------------------------------

            # coordinates = tf.transpose(coordinates, perm=(1, 0, 2))
            # import sys; sys.exit()

            return coordinates



# ==========================================================================
class Encoder_2(tf.keras.Model):
    def __init__(self):
        super(Encoder_2, self).__init__()

        self.lstm_forward  = layers.LSTM(lstm_cells, return_sequences=True, return_state=True, dropout=0.5)
        self.lstm_backward = layers.LSTM(lstm_cells, return_sequences=True, return_state=True, dropout=0.5, go_backwards=True)

    def call(self, x, training=True):
        # output, hidden state, cell state
        H1, h1, c1 = self.lstm_forward(x)
        H2, h2, c2 = self.lstm_backward(x)
        H = tf.concat([H1, H2], axis=2)     # (batch, N, 1600)
        h = tf.concat([h1, h2], axis=1)     # (batch, 1600)
        c = tf.concat([c1, c2], axis=1)     # (batch, 1600)

        return H, h, c

class Decoder_2(tf.keras.Model):
    def __init__(self):
        super(Decoder_2, self).__init__()

        self.lstm = layers.LSTM(lstm_cells * 2, return_sequences=True, return_state=True, dropout=0.5)
        self.attention = layers.Attention()
        self.dense1 = layers.Dense(600, name='flatten')
        self.dense2 = layers.Dense(3)

    def call(self, inputs, training=True):
        y, s0, c0, H_value = inputs

        S, h, c = self.lstm(y, initial_state=[s0, c0])  # S : all hidden states for y.
        S_key = tf.concat([s0[:, tf.newaxis, :], S[:, :-1, :]], axis=1) # --> Key of Attention
        A = self.attention([S_key, H_value])    # key, value

        x = tf.concat([S, A], axis=-1) #  (32, N*3, 3200)
        x = self.dense1(x)
        y = self.dense2(x)

        return y, h, c

class NBCC_RGN_Attention_2(models.Model):

    def __init__(self, batch_size):
        super(NBCC_RGN_Attention_2, self).__init__()

        self.batch_size = batch_size

        # Attention Network
        self.enc = Encoder_2()
        self.dec = Decoder_2()

    def call(self, inputs, training=False, mask=None):

        if training is True:
            x, y = inputs

            H, h, c = self.enc(x)   # output, hidden state, cell state
            coordinates, _, _ = self.dec((y, h, c, H))
            coordinates = tf.transpose(coordinates, perm=(1, 0, 2))
            return coordinates

        else:
            x = inputs
            coord_len = x.shape[1]

            H, h, c = self.enc(x)

            # -------------------------------------------------------------------------
            # ver. 1
            # test_batch_size = 1
            # y = tf.ones(shape=[1, 3, 3], dtype=tf.float32) # one residue : N-CA-C * 3
            #
            # for idx in tf.range(coord_len):
            #     y, h, c = self.dec((y, h, c, H)) # (3, 1, 3)
            #
            #     if idx == 0:
            #         coordinates = tf.identity(y)
            #     else:
            #         coordinates = tf.concat([coordinates, y], axis=1)


            # -------------------------------------------------------------------------
            # ver. 2
            y = tf.ones(shape=[1, coord_len*3, 3], dtype=tf.float32)
            coordinates, _, _ = self.dec((y, h, c, H))

            # -------------------------------------------------------------------------

            coordinates = tf.transpose(coordinates, perm=(1, 0, 2))

            return coordinates
# ==========================================================================

class Encoder_3(tf.keras.Model):
    def __init__(self):
        super(Encoder_3, self).__init__()
        self.lstm = layers.LSTM(lstm_cells, return_sequences=True, return_state=True, dropout=0.5)

    def call(self, x, training=True):
        # output, hidden state, cell state
        _, h, c = self.lstm(x)
        return h, c

class Decoder_3(tf.keras.Model):
    def __init__(self):
        super(Decoder_3, self).__init__()
        self.lstm = layers.LSTM(lstm_cells, return_sequences=True, return_state=True, dropout=0.5)
        # dihedrals
        self.dense1 = layers.Dense(400)
        self.dense2 = layers.Dense(3) # activation='softmax'

    def call(self, inputs, training=True):
        x, h, c = inputs
        x, h, c = self.lstm(x, initial_state=[h, c])

        x = self.dense1(x)
        y_ = self.dense2(x)                   # 32, 1977, 60 : 1977 --> 659

        return y_, h, c


class NBCC_RGN_Attention_3(models.Model):

    def __init__(self, batch_size):
        super(NBCC_RGN_Attention_3, self).__init__()

        self.batch_size = batch_size

        # Attention Network
        self.enc = Encoder_3()
        self.dec = Decoder_3()

    def call(self, inputs, training=False, mask=None):
        # N = residues * 3
        if training is True:
            x, y = inputs
            # x = tf.transpose(x, perm=(1, 0, 2))
            # y = tf.transpose(y, perm=(1, 0, 2))
            # print('x:', x.shape)
            # print('y:', y.shape)

            h, c = self.enc(x)   # output, hidden state, cell state : (batch, 800)

            coordinates, _, _ = self.dec((y, h, c))

            coordinates = tf.transpose(coordinates, perm=(1, 0, 2))

            return coordinates

        else:
            x = inputs
            x = tf.transpose(x, perm=(1, 0, 2))     # (batch, N, 62)

            h, c = self.enc(x)

            # -------------------------------------------------------------------------
            # ver. 1
            y = tf.zeros(shape=[32, 1, 3], dtype=tf.float32) # (batch, N * 3, 3) : N-CA-C

            coord_len = x.shape[1] * NUM_DIMENSIONS
            for idx in range(coord_len):
                # y를 계속 다음 입력으로 사용
                y, h, c = self.dec((y, h, c))

                if idx == 0:
                    coordinates = tf.identity(y)
                else:
                    coordinates = tf.concat([coordinates, y], axis=1)

            return coordinates

# ==========================================================================
class Encoder5(tf.keras.Model):
    def __init__(self):
        super(Encoder5, self).__init__()

        # self.dense = layers.Dense(1)
        # self.emb = layers.Embedding(input_dim=700, output_dim=700)

        self.lstm = layers.LSTM(lstm_cells, return_sequences=True, return_state=True, dropout=0.5)

        # self.lstm_forward  = layers.LSTM(lstm_cells, return_sequences=True, return_state=True, dropout=0.5)
        # self.lstm_backward = layers.LSTM(lstm_cells, return_sequences=True, return_state=True, dropout=0.5, go_backwards=True)

    def call(self, x, training=True):
        # print('encoder x :', x.shape)

        y, h, c = self.lstm(x)

        # y1, h1, c1 = self.lstm_forward(x)
        # y2, h2, c2 = self.lstm_backward(x)
        # y = tf.concat([y1, y2], axis=2)     # (batch, N, 1600)
        # h = tf.concat([h1, h2], axis=1)     # (batch, 1600)
        # c = tf.concat([c1, c2], axis=1)     # (batch, 1600)

        return y, h, c

class Decoder5(tf.keras.Model):
    def __init__(self):
        super(Decoder5, self).__init__()
        self.lstm = layers.LSTM(lstm_cells, return_sequences=True, return_state=True, dropout=0.5)
        self.dense1 = layers.Dense(600)
        self.dense2 = layers.Dense(60)
        # self.dense2 = layers.Dense(9, activation='softmax')

    def call(self, inputs, training=True):
        x, h, c = inputs
        x, h, c = self.lstm(x, initial_state=[h, c])

        x = self.dense1(x)
        y_ = self.dense2(x)                   # 32, 1977, 60 : 1977 --> 659

        return y_, h, c

class NBCC_RGN_seq2seq(models.Model):

    def __init__(self, batch_size):
        super(NBCC_RGN_seq2seq, self).__init__()

        self.batch_size = batch_size

        # alphabet
        alphabet_initializer = tf.random_uniform_initializer(minval=-3.14159, maxval=3.14159)
        self.alphabets = self.add_weight(shape=(alphabet_size, NUM_DIHEDRALS), initializer=alphabet_initializer, trainable=True, name='alphabets')

        # dihedrals to coordinates
        self.softmax = layers.Softmax(name='softmax')
        self.coordinate_layer = CoordinateLayer(max_seq_length) # initialize by max_seq_length

        # Attention Network
        self.enc = Encoder5()
        self.dec = Decoder5()

        self.dense = layers.Dense(1)
        self.emb = layers.Embedding(10000, 700)

    def call(self, inputs, training=False, mask=None):
        # N = residues * 3
        if training is True:
            x, y = inputs

            x = tf.squeeze(self.dense(x))
            x = self.emb(x)

            _, h, c = self.enc(x)   # output, hidden state, cell state

            # -----------------
            # coordinates, _, _ = self.dec((y, h, c))  # !!!

            # -----------------
            x, _, _ = self.dec((y, h, c))
            flat_x = tf.reshape(x, [-1, 60])            # (N x batch_size, 60)
            x = self.softmax(flat_x)                    # (N x batch_size, 60)
            flat_dihedrals = reduce_mean_angle(x, self.alphabets) # (N x batch_size, 60) * (60, 3) = (N x batch_size, 3)
            batch_size = self.batch_size
            coordinates = self.coordinate_layer(flat_dihedrals, batch_size)

            return coordinates

        else:
            x = inputs
            coord_len = x.shape[1]

            x = self.dense(x)
            x = tf.reshape(x, shape=[1, coord_len])
            # print('x1 :', x.shape)
            x = self.emb(x)
            # print('x2 :', x.shape)

            _, h, c = self.enc(x)

            # -----------------------------------------------------------------
            # test_batch_size = 1
            # y = tf.ones(shape=[1, 1, 9], dtype=tf.float32) # (batch, N * 3, 3) : N-CA-C
            # for idx in range(coord_len):
            #     # y를 계속 다음 입력으로 사용
            #     y, h, c = self.dec((y, h, c))
            #
            #     if idx == 0:
            #         coordinates = tf.identity(y)
            #     else:
            #         coordinates = tf.concat([coordinates, y], axis=1)

            # -----------------------------------------------------------------
            y = tf.ones(shape=[1, coord_len, 9], dtype=tf.float32)
            x, _, _ = self.dec((y, h, c))
            flat_x = tf.reshape(x, [-1, 60])            # (N x batch_size, 60)
            x = self.softmax(flat_x)                    # (N x batch_size, 60)
            flat_dihedrals = reduce_mean_angle(x, self.alphabets) # (N x batch_size, 60) * (60, 3) = (N x batch_size, 3)
            batch_size = self.batch_size
            coordinates = self.coordinate_layer(flat_dihedrals, batch_size)

            return coordinates


# ==========================================================================
