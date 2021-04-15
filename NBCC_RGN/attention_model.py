"""
Attention network model (with LSTM)
"""

import tensorflow as tf
from geom_ops import reduce_mean_angle, dihedral_to_point, point_to_coordinate
from tensorflow.keras import layers, models, losses, optimizers, metrics, losses

NUM_DIHEDRALS = 3
NUM_DIMENSIONS = 3

lstm_cells = 800
max_seq_length = 700
alphabet_size = 60

# ==========================================================================
def adam_optimizer():
    optimizer = optimizers.Adam(learning_rate=0.0001, beta_1=0.95, beta_2=0.99, epsilon=1e-07)
    return optimizer


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
    def __init__(self):
        super(Decoder, self).__init__()

        self.lstm = layers.LSTM(lstm_cells * 2, return_sequences=True, return_state=True, dropout=0.5)

        # Attention layer !!!
        self.attention = layers.Attention()

        self.dense1 = layers.Dense(60, name='flatten')
        self.dense2 = layers.Dense(3)

    def call(self, inputs, training=True):
        # y  : ground truth                 (batch, N*3 , 3)
        # s0 : hidden state of Encoder      (batch, 1600)
        # c0 : cell state of Encoder        (batch, 1600)
        # H  : output of Encoder            (batch, N, 1600)
        y, s0, c0, H_value = inputs

        S, h, c = self.lstm(y, initial_state=[s0, c0])  # S : all hidden states for y. (batch, N, 1600)

        # print('S :', S.shape)       # (32, N*3 , 3)
        # print('h :', s0.shape)      # (32, 1600)
        # print('c :', c0.shape)

        S_key = tf.concat([s0[:, tf.newaxis, :], S[:, :-1, :]], axis=1) # --> Key of Attention
        # print('S_key :', S_key.shape)
        # H_value : output of Encoder --> Value of Attention

        A = self.attention([S_key, H_value])    # key, value
        # print('A :', A.shape)

        y_ = tf.concat([S, A], axis=-1)
        # print('y_ :', y_.shape)                     #  (32, N*3, 3200)

        x = self.dense1(y_)                          # (32, N, 60)
        # print('dense1 :', x.shape)

        coordinates = self.dense2(x)                          # (N, batch_size, 3)
        # print('coordinates :', coordinates.shape)

        return coordinates, h, c


class NBCC_RGN_Attention(models.Model):

    def __init__(self, batch_size):
        super(NBCC_RGN_Attention, self).__init__()

        self.batch_size = batch_size

        # Attention Network
        self.enc = Encoder()
        self.dec = Decoder()
        self.sos = 0.0


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
            coordinates = tf.transpose(coordinates, perm=(1, 0, 2))

            return coordinates

        else:
            x = inputs
            x = tf.transpose(x, perm=(1, 0, 2))     # (batch, N, 62)

            H, h, c = self.enc(x)

            # -------------------------------------------------------------------------
            # ver. 1 - node26
            y = tf.ones(shape=[32, 3, 3], dtype=tf.float32) # (batch, N * 3, 3) : N-CA-C

            coord_len = x.shape[1] # * NUM_DIMENSIONS
            for idx in range(coord_len):
                # y를 계속 다음 입력으로 사용
                y, h, c = self.dec((y, h, c, H))

                # a = tf.cast(tf.argmax(y, axis=-1), dtype=tf.float32)
                # print('a:', a.shape)

                if idx == 0:
                    coordinates = tf.identity(y)
                else:
                    coordinates = tf.concat([coordinates, y], axis=1)

            # -------------------------------------------------------------------------
            # ver. 2 - node27
            # coord_len = x.shape[1] * NUM_DIMENSIONS
            # y = tf.ones(shape=[32, coord_len, NUM_DIHEDRALS], dtype=tf.float32)
            # coordinates = self.dec((y, h, c, H))

            # -------------------------------------------------------------------------
            # print(coordinates.shape)

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
        self.dense = layers.Dense(3) # activation='softmax'

    def call(self, inputs, training=True):
        x, h, c = inputs
        x, h, c = self.lstm(x, initial_state=[h, c])

        y_ = self.dense(x)                   # 32, 1977, 60 : 1977 --> 659

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
            # print('y_ :', coordinates.shape)

            coordinates = tf.transpose(coordinates, perm=(1, 0, 2))

            return coordinates

        else:
            x = inputs
            x = tf.transpose(x, perm=(1, 0, 2))     # (batch, N, 62)

            h, c = self.enc(x)

            # -------------------------------------------------------------------------
            # ver. 1 - node26
            y = tf.zeros(shape=[32, 1, 3], dtype=tf.float32) # (batch, N * 3, 3) : N-CA-C

            coord_len = x.shape[1] * NUM_DIMENSIONS
            for idx in range(coord_len):
                # y를 계속 다음 입력으로 사용
                y, h, c = self.dec((y, h, c))

                if idx == 0:
                    coordinates = tf.identity(y)
                else:
                    coordinates = tf.concat([coordinates, y], axis=1)

            coordinates = tf.transpose(coordinates, perm=(1, 0, 2))

            return coordinates

# ==========================================================================
