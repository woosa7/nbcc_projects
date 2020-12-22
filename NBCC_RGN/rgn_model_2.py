import glob
import numpy as np
import collections
import tensorflow as tf
from tensorflow.keras import layers, models

"""
python 3.7
tensorflow 2.3
"""

NUM_DIHEDRALS = 3
NUM_DIMENSIONS = 3

max_length = 700
batch_size = 32

tf.random.set_seed(100)

# --------------------------------------------------------------------------
def reduce_mean_angle(weights, angles):
    """
        weights: [BATCH_SIZE, NUM_ANGLES]
        angles:  [NUM_ANGLES, NUM_DIHEDRALS]
        Returns: [BATCH_SIZE, NUM_DIHEDRALS]
    """
    weights = tf.convert_to_tensor(weights, name='weights')
    angles  = tf.convert_to_tensor(angles,  name='angles')

    # use real-numbered pairs of values
    sins = tf.sin(angles)
    coss = tf.cos(angles)

    y_coords = tf.matmul(weights, sins)
    x_coords = tf.matmul(weights, coss)

    return tf.atan2(y_coords, x_coords)


def dihedral_to_point(dihedral):
    """ Takes triplets of dihedral angles (omega, phi, psi) and returns 3D points ready for use in reconstruction of coordinates.

        dihedral: [NUM_STEPS, BATCH_SIZE, NUM_DIHEDRALS]
        Returns:  [NUM_STEPS x NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMENSIONS]
    """
    # Bond lengths and angles are based on idealized averages.
    r     = np.array([145.801, 152.326, 132.868], dtype='float32')      # BOND_LENGTHS
    theta = np.array([  2.124,   1.941,   2.028], dtype='float32')      # BOND_ANGLES

    dihedral = tf.convert_to_tensor(dihedral) # [NUM_STEPS, BATCH_SIZE, NUM_DIHEDRALS]

    num_steps  = tf.shape(dihedral)[0]
    batch_size = dihedral.get_shape().as_list()[1] # important to use get_shape() to keep batch_size fixed for performance reasons

    r_cos_theta = tf.constant(r * np.cos(np.pi - theta), name='r_cos_theta') # [NUM_DIHEDRALS]
    r_sin_theta = tf.constant(r * np.sin(np.pi - theta), name='r_sin_theta') # [NUM_DIHEDRALS]

    pt_x = tf.tile(tf.reshape(r_cos_theta, [1, 1, -1]), [num_steps, batch_size, 1], name='pt_x') # [NUM_STEPS, BATCH_SIZE, NUM_DIHEDRALS]
    pt_y = tf.multiply(tf.cos(dihedral), r_sin_theta,                               name='pt_y') # [NUM_STEPS, BATCH_SIZE, NUM_DIHEDRALS]
    pt_z = tf.multiply(tf.sin(dihedral), r_sin_theta,                               name='pt_z') # [NUM_STEPS, BATCH_SIZE, NUM_DIHEDRALS]

    pt = tf.stack([pt_x, pt_y, pt_z])                                                       # [NUM_DIMS, NUM_STEPS, BATCH_SIZE, NUM_DIHEDRALS]
    pt_perm = tf.transpose(pt, perm=[1, 3, 2, 0])                                           # [NUM_STEPS, NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMS]
    pt_final = tf.reshape(pt_perm, [num_steps * NUM_DIHEDRALS, batch_size, NUM_DIMENSIONS]) # [NUM_STEPS x NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMS]

    return pt_final


def point_to_coordinate(pt, num_fragments=6, name=None):
    """ Takes points from dihedral_to_point and sequentially converts them into the coordinates of a 3D structure.
    """

    with tf.name_scope(name='point_to_coordinate') as scope:
        pt = tf.convert_to_tensor(pt, name='pt')
        # print('pt 1 :', pt.shape)       # (None, 32, 3)

        s = tf.shape(pt)[0]     # NUM_STEPS x NUM_DIHEDRALS

        # initial three coordinates (specifically chosen to eliminate need for extraneous matmul)
        Triplet = collections.namedtuple('Triplet', 'a, b, c')

        batch_size = pt.get_shape().as_list()[1] # BATCH_SIZE
        init_mat = np.array([[-np.sqrt(1.0 / 2.0), np.sqrt(3.0 / 2.0), 0], [-np.sqrt(2.0), 0, 0], [0, 0, 0]], dtype='float32')
        init_coords = Triplet(*[tf.reshape(tf.tile(row[np.newaxis], tf.stack([num_fragments * batch_size, 1])),
                                           [num_fragments, batch_size, NUM_DIMENSIONS]) for row in init_mat])
                                           # 3 * [6, 32, 3]
                                           # NUM_DIHEDRALS x [NUM_FRAGS, BATCH_SIZE, NUM_DIMENSIONS]

        # pad points to yield equal-sized fragments
        r = ((num_fragments - (s % num_fragments)) % num_fragments)          # (NUM_FRAGS x FRAG_SIZE) - (NUM_STEPS x NUM_DIHEDRALS)
        pt = tf.pad(pt, [[0, r], [0, 0], [0, 0]])                            # [NUM_FRAGS x FRAG_SIZE, BATCH_SIZE, NUM_DIMENSIONS]
        pt = tf.reshape(pt, [num_fragments, -1, batch_size, NUM_DIMENSIONS]) # [NUM_FRAGS, FRAG_SIZE,  BATCH_SIZE, NUM_DIMENSIONS]
        pt = tf.transpose(pt, perm=[1, 0, 2, 3])                             # [FRAG_SIZE, NUM_FRAGS,  BATCH_SIZE, NUM_DIMENSIONS]

        # print('pt', pt.shape)     # (None, 6, 32, 3)

        # extension function used for single atom reconstruction and whole fragment alignment
        def extend(tri, pt, multi_m):
            """
            Args:
                tri: NUM_DIHEDRALS x [NUM_FRAGS/0,         BATCH_SIZE, NUM_DIMENSIONS]
                pt:                  [NUM_FRAGS/FRAG_SIZE, BATCH_SIZE, NUM_DIMENSIONS]
                multi_m: bool indicating whether m (and tri) is higher rank. pt is always higher rank; what changes is what the first rank is.
            """
            bc = tf.nn.l2_normalize(tri.c - tri.b, -1, name='bc')                                        # [NUM_FRAGS/0, BATCH_SIZE, NUM_DIMS]
            n = tf.nn.l2_normalize(tf.linalg.cross(tri.b - tri.a, bc), -1, name='n')                            # [NUM_FRAGS/0, BATCH_SIZE, NUM_DIMS]
            if multi_m: # multiple fragments, one atom at a time.
                m = tf.transpose(tf.stack([bc, tf.linalg.cross(n, bc), n]), perm=[1, 2, 3, 0], name='m')        # [NUM_FRAGS,   BATCH_SIZE, NUM_DIMS, 3 TRANS]
            else: # single fragment, reconstructed entirely at once.
                s = tf.pad(tf.shape(pt), [[0, 1]], constant_values=3)                                    # FRAG_SIZE, BATCH_SIZE, NUM_DIMS, 3 TRANS
                m = tf.transpose(tf.stack([bc, tf.linalg.cross(n, bc), n]), perm=[1, 2, 0])                     # [BATCH_SIZE, NUM_DIMS, 3 TRANS]
                m = tf.reshape(tf.tile(m, [s[0], 1, 1]), s, name='m')                                    # [FRAG_SIZE, BATCH_SIZE, NUM_DIMS, 3 TRANS]
            coord = tf.add(tf.squeeze(tf.matmul(m, tf.expand_dims(pt, 3)), axis=3), tri.c, name='coord') # [NUM_FRAGS/FRAG_SIZE, BATCH_SIZE, NUM_DIMS]
            return coord

        # loop over FRAG_SIZE in NUM_FRAGS parallel fragments, sequentially generating the coordinates for each fragment across all batches
        i = tf.constant(0)
        s_padded = tf.shape(pt)[0]
        # print('s_padded :', s_padded)

        coords_ta = tf.TensorArray(tf.float32, size=s_padded)

        def loop_extend(i, tri, coords_ta): # FRAG_SIZE x [NUM_FRAGS, BATCH_SIZE, NUM_DIMENSIONS]
            coord = extend(tri, pt[i], True)
            return [i + 1, Triplet(tri.b, tri.c, coord), coords_ta.write(i, coord)]


        _, tris, coords_pretrans_ta = tf.while_loop(lambda i, _1, _2: i < s_padded, loop_extend, [i, init_coords, coords_ta])
                                      # NUM_DIHEDRALS x [NUM_FRAGS, BATCH_SIZE, NUM_DIMENSIONS],
                                      # FRAG_SIZE x [NUM_FRAGS, BATCH_SIZE, NUM_DIMENSIONS]

        # loop over NUM_FRAGS in reverse order, bringing all the downstream fragments in alignment with current fragment
        coords_pretrans = tf.transpose(coords_pretrans_ta.stack(), perm=[1, 0, 2, 3]) # [NUM_FRAGS, FRAG_SIZE, BATCH_SIZE, NUM_DIMENSIONS]

        i = tf.shape(coords_pretrans)[0] # NUM_FRAGS

        def loop_trans(i, coords):
            transformed_coords = extend(Triplet(*[di[i] for di in tris]), coords, False)
            return [i - 1, tf.concat([coords_pretrans[i], transformed_coords], 0)]

        _, coords_trans = tf.while_loop(lambda i, _: i > -1, loop_trans, [i - 2, coords_pretrans[-1]])
                          # [NUM_FRAGS x FRAG_SIZE, BATCH_SIZE, NUM_DIMENSIONS]
        # print('coords_trans :', coords_trans.shape)       # (None, 32, 3)

        # lose last atom and pad from the front to gain an atom ([0,0,0], consistent with init_mat), to maintain correct atom ordering
        coords = tf.pad(coords_trans[:s-1], [[1, 0], [0, 0], [0, 0]]) # [NUM_STEPS x NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMENSIONS]
        # print('coords :', coords.shape)   # (None, 32, 3)

        return coords

# --------------------------------------------------------------------------

# ==========================================================================
# training dataset
# all       : 42528 = 32 * 1329
# less 700  : 41789 = 32 * 1306

"""
Data Loader
"""
def mapping_func(serialized_examples):
    parsed_examples = tf.io.parse_sequence_example(serialized_examples,
                            context_features={'id':         tf.io.RaggedFeature(tf.string)},
                            sequence_features={
                                            'primary':      tf.io.RaggedFeature(tf.int64),
                                            'evolutionary': tf.io.RaggedFeature(tf.float32),
                                            'secondary':    tf.io.RaggedFeature(tf.int64),
                                            'tertiary':     tf.io.RaggedFeature(tf.float32),
                                            'mask':         tf.io.RaggedFeature(tf.float32)}
                                            )

    ids = parsed_examples[0]['id']
    features = parsed_examples[1]

    return features, ids

def filter_func(features, ids):
    primary = features['primary']

    pri_length = tf.size(primary)
    keep = pri_length <= max_length

    return keep

def get_data(features):
    # int64 to int32
    primary = features['primary']
    primary = tf.dtypes.cast(primary, tf.int32)
    evolutionary = features['evolutionary']
    tertiary = features['tertiary']

    # convert to one_hot_encoding
    primary = tf.squeeze(primary, axis=2)
    one_hot_primary = tf.one_hot(primary, 20)

    # padding
    one_hot_primary = tf.RaggedTensor.to_tensor(one_hot_primary)
    evolutionary = tf.RaggedTensor.to_tensor(evolutionary)
    tertiary = tf.RaggedTensor.to_tensor(tertiary)

    # (batch_size, N, 20) to (N, batch_size, 20)
    one_hot_primary = tf.transpose(one_hot_primary, perm=(1, 0, 2))
    evolutionary = tf.transpose(evolutionary, perm=(1, 0, 2))
    tertiary = tf.transpose(tertiary, perm=(1, 0, 2))

    inputs = tf.concat((one_hot_primary, evolutionary), axis=2)

    # TODO
    y_len = inputs.shape[0]
    ter_y = tertiary[:y_len]

    return inputs, ter_y


"""
Data Generator
"""
phase = 'training'
file_list = glob.glob('./RGN11/data/ProteinNet11/{}/*'.format(phase))

train_dataset = tf.data.TFRecordDataset(tf.data.Dataset.list_files(file_list, shuffle=True)).map(mapping_func).filter(filter_func).batch(batch_size, drop_remainder=True).prefetch(1)

def generator():
    for element in train_dataset:
        features = element[0]
        ids = element[1]
        inputs, ter_y = get_data(features)
        yield inputs, ter_y #, ids


# train_data = tf.data.Dataset.from_generator(generator, output_types=(tf.float32, tf.float32, tf.string))
train_data = tf.data.Dataset.from_generator(generator, output_types=(tf.float32, tf.float32))

# ==========================================================================
def get_model():
    alphabets = tf.random.uniform(shape=[60,3], minval=-3.14, maxval=3.14)

    input_x = layers.Input(shape=(batch_size, 62))  # (N, 32, 62)
    # -------------------------------------------------
    # Bi-directional LSTM
    out_x = layers.Bidirectional(layers.LSTM(800, time_major=True, return_sequences=True, dropout=0.5), name='bi_lstm1')(input_x)
    out_x = layers.Bidirectional(layers.LSTM(800, time_major=True, return_sequences=True, dropout=0.5), name='bi_lstm2')(out_x)
    # RNN output : (N, 32, 1600)
    # -------------------------------------------------
    # dihedrals
    out_x = layers.Dense(60, name='flatten')(out_x)             # (N, 32, 60)
    num_steps = tf.shape(out_x)[0]

    flat_out_x = tf.reshape(out_x, [-1, 60])
    flat_out_x = layers.Softmax(name='softmax')(flat_out_x)     # (N, 60)
    flat_dihedrals = reduce_mean_angle(flat_out_x, alphabets)     # (N, 60) * (60, 3) = (N, 3)
    dihedrals = tf.reshape(flat_dihedrals, [num_steps, batch_size, 3])  # (None, 32, 3)

    # print(dihedrals.shape)

    # -------------------------------------------------
    # TODO
    model = models.Model(input_x, dihedrals)
    model.summary()

    return model


model = get_model()
model.compile(optimizer='adam', loss='mse', metrics=['mean_squared_error'], experimental_run_tf_function=False)

history = model.fit(train_data, epochs=2, verbose=1)

# -------------------------------------------------------------
