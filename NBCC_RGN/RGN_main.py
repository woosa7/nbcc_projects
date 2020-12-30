import glob
import numpy as np
import collections
import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers, metrics
from geom_ops import reduce_mean_angle, dihedral_to_point, point_to_coordinate, drmsd

"""
python 3.7
tensorflow 2.3
"""

NUM_DIHEDRALS = 3
NUM_DIMENSIONS = 3

batch_size = 32

max_seq_length = 700
alphabet_size = 60
num_edge_residues = 0

include_loss = True
loss_atoms = 'c_alpha'
tertiary_weight = tf.Variable(1.0)
tertiary_normalization = 'first'
batch_dependent_normalization = True

num_evaluation_invocations = 1 # simple loss

tf.random.set_seed(100)

# ==========================================================================
def masking_matrix(mask, name=None):
    """ Constructs a masking matrix to zero out pairwise distances due to missing residues or padding.
        This function needs to be called for each individual sequence, and so it's folded in the reading/queuing pipeline for performance reasons.

    Args:
        mask: 0/1 vector indicating whether a position should be masked (0) or not (1)

    Returns:
        A square matrix with all 1s except for rows and cols whose corresponding indices in mask are set to 0.
        [MAX_SEQ_LENGTH, MAX_SEQ_LENGTH]
    """

    with tf.name_scope('masking_matrix') as scope:
        mask = tf.convert_to_tensor(mask, name='mask')
        mask = tf.expand_dims(mask, 0)
        base = tf.ones([tf.size(mask), tf.size(mask)])
        matrix_mask = base * mask * tf.transpose(mask)

        return matrix_mask

def weighting_matrix(weights, name=None):
    """ Takes a vector of weights and returns a weighting matrix in which the ith weight is in the ith upper diagonal of the matrix. All other entries are 0.
        This functions needs to be called once per curriculum update / iteration, but then used for the entire batch.

        This function intimately mixes python and TF code. It can do so because all the python code
        needs to be run only once during the initial construction phase and does not rely on any
        tensor values. This interaction is subtle however.

    Args:
        weights: Curriculum weights. A TF tensor that is expected to change as curriculum progresses. [MAX_SEQ_LENGTH - 1]

    Returns
        [MAX_SEQ_LENGTH, MAX_SEQ_LENGTH]
    """

    with tf.name_scope('weighting_matrix') as scope:
        weights = tf.convert_to_tensor(weights, name='weights')

        max_seq_length = weights.get_shape().as_list()[0] + 1
        split_indices = np.diag_indices(max_seq_length)

        flat_indices = []
        flat_weights = []
        for i in range(max_seq_length - 1):
            indices_subset = np.concatenate((split_indices[0][:-(i+1), np.newaxis], split_indices[1][i+1:, np.newaxis]), 1)
            weights_subset = tf.fill([len(indices_subset)], weights[i])
            flat_indices.append(indices_subset)
            flat_weights.append(weights_subset)
        flat_indices = np.concatenate(flat_indices)
        flat_weights = tf.concat(flat_weights, 0)

        # NBCC modified for tf 2.0
        # tf 1.0 : sparse_indices, output_shape, sparse_values
        # mat = tf.sparse_to_dense(flat_indices, [max_seq_length, max_seq_length], flat_weights, validate_indices=False, name=scope)

        # indices, values, dense_shape
        mat_st = tf.sparse.SparseTensor(flat_indices, flat_weights, [max_seq_length, max_seq_length])
        mat = tf.sparse.to_dense(mat_st, validate_indices=False)

        return mat

def _weights(masks):
    """ Returns dRMSD weights that mask meaningless (missing or longer than sequence residues) pairwise distances and incorporate the state of
        the curriculum to differentially weigh pairwise distances based on their proximity.
        num_edge_residues = 0
    """

    # loss_atoms = 'c_alpha' & config['mode'] = None:
    # no loss-based curriculum, create fixed weighting matrix that weighs all distances equally.
    # minus one factor is there because we ignore self-distances.
    flat_curriculum_weights = np.ones(max_seq_length - num_edge_residues - 1, dtype='float32')

    # weighting matrix for entire batch that accounts for curriculum weighting.
    unnormalized_weights = weighting_matrix(flat_curriculum_weights, name='unnormalized_weights')
                           # [NUM_STEPS - NUM_EDGE_RESIDUES, NUM_STEPS - NUM_EDGE_RESIDUES]

    # create final weights by multiplying with masks and normalizing.
    mask_length = tf.shape(masks)[0]
    unnormalized_masked_weights = masks * unnormalized_weights[:mask_length, :mask_length, tf.newaxis]

    masked_weights = tf.math.divide(unnormalized_masked_weights,
                                    tf.reduce_sum(unnormalized_masked_weights, axis=[0, 1]),
                                    name='weights')

    return masked_weights, flat_curriculum_weights

def _drmsds(coordinates, targets, weights):
    """ Computes reduced weighted dRMSD loss (as specified by weights)
        between predicted tertiary structures and targets. """

    # lose end residues if desired
    if num_edge_residues > 0:
        coordinates = coordinates[:-(num_edge_residues * NUM_DIHEDRALS)]

    # if only c_alpha atoms are requested then subsample
    if loss_atoms == 'c_alpha': # starts at 1 because c_alpha atoms are the second atoms
        coordinates = coordinates[1::NUM_DIHEDRALS] # [NUM_STEPS - NUM_EDGE_RESIDUES, BATCH_SIZE, NUM_DIMENSIONS]
        targets     =     targets[1::NUM_DIHEDRALS] # [NUM_STEPS - NUM_EDGE_RESIDUES, BATCH_SIZE, NUM_DIMENSIONS]

    # compute per structure dRMSDs
    drmsds = drmsd(coordinates, targets, weights, name='drmsds') # [BATCH_SIZE]

    # add to relevant collections for summaries, etc.
    # if config['log_model_summaries']:
    #     tf.add_to_collection(config['name'] + '_drmsdss', drmsds)

    return drmsds

def effective_steps(masks, num_edge_residues, name=None):
    """ Returns the effective number of steps, i.e. number of residues that are non-missing and are not just padding, given a masking matrix.

    Args:
        masks: A batch of square masking matrices (batch is last dimension)
        [MAX_SEQ_LENGTH, MAX_SEQ_LENGTH, BATCH_SIZE]

    Returns:
        A vector with the effective number of steps
        [BATCH_SIZE]
    """

    with tf.name_scope('effective_steps') as scope:
        masks = tf.convert_to_tensor(masks, name='masks')

        # NBCC modified for tf 2.0
        # traces = tf.matrix_diag_part(tf.transpose(masks, [2, 0, 1]))
        traces = tf.linalg.diag_part(tf.transpose(masks, [2, 0, 1]))
        eff_stepss = tf.add(tf.reduce_sum(traces, [1]), num_edge_residues, name=scope) # NUM_EDGE_RESIDUES shouldn't be here, but I'm keeping it for
                                                                                       # legacy reasons. Just be clear that it's _always_ wrong to have
                                                                                       # it here, even when it's not equal to 0.
        return eff_stepss

@tf.function(experimental_relax_shapes=True)
def tertiary_loss(losses, masks):
    """ Reduces loss according to normalization order.
        num_evaluation_invocations = 1 (simple loss)
        this = _reduce_loss_quotient() + _accumulate_loss()
    """
    # NBCC modified for tf 2.0
    filters = {'all': tf.tile([True], [batch_size,])}  # training mode
    group_filter = filters['all']

    losses_filtered = tf.boolean_mask(losses, group_filter) # it will give problematic results if all entries are removed

    if tertiary_normalization == 'zeroth':
        loss_factors = tf.ones_like(losses_filtered)
    elif tertiary_normalization == 'first':
        loss_factors = tf.boolean_mask(effective_steps(masks, num_edge_residues), group_filter)
        fixed_denominator_factor = float(max_seq_length - num_edge_residues)
    elif tertiary_normalization == 'second':
        eff_num_stepss = tf.boolean_mask(effective_steps(masks, num_edge_residues), group_filter)
        loss_factors = (tf.square(eff_num_stepss) - eff_num_stepss) / 2.0
        fixed_denominator_factor = float(max_seq_length - num_edge_residues)
        fixed_denominator_factor = ((fixed_denominator_factor ** 2) - fixed_denominator_factor) / 2.0

    numerator = tf.reduce_sum(loss_factors * losses_filtered)

    if batch_dependent_normalization or tertiary_normalization == 'zeroth':
        denominator = tf.reduce_sum(loss_factors)
    else:
        denominator = tf.multiply(tf.cast(tf.size(loss_factors), tf.float32), fixed_denominator_factor)

    # NBCC modified for tf 2.0
    # _accumulate_loss() --> simple loss
    tertiary_loss = tf.divide(numerator, denominator) * tertiary_weight

    return tertiary_loss

# ==========================================================================
"""
Data Loader

training dataset
all       : 42528 / 32 = 1329
less 700  : 41789 / 32 = 1305
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
    keep = pri_length <= max_seq_length

    return keep

def get_data(features):
    # int64 to int32
    primary = features['primary']
    primary = tf.dtypes.cast(primary, tf.int32)
    evolutionary = features['evolutionary']
    tertiary = features['tertiary']
    mask = features['mask']

    # convert to one_hot_encoding
    primary = tf.squeeze(primary, axis=2)
    one_hot_primary = tf.one_hot(primary, 20)

    # padding
    one_hot_primary = tf.RaggedTensor.to_tensor(one_hot_primary)
    evolutionary = tf.RaggedTensor.to_tensor(evolutionary)
    tertiary = tf.RaggedTensor.to_tensor(tertiary)
    mask = tf.RaggedTensor.to_tensor(mask)
    if tf.shape(mask)[1] == 0:
        raise RuntimeError('mask size is 0')

    # (batch_size, N, 20) to (N, batch_size, 20)
    one_hot_primary = tf.transpose(one_hot_primary, perm=(1, 0, 2))
    evolutionary = tf.transpose(evolutionary, perm=(1, 0, 2))
    tertiary = tf.transpose(tertiary, perm=(1, 0, 2))

    inputs = tf.concat((one_hot_primary, evolutionary), axis=2)

    return inputs, tertiary, mask

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
        inputs, tertiaries, ter_masks = get_data(features)
        yield inputs, tertiaries, ter_masks, ids

train_data = tf.data.Dataset.from_generator(generator, output_types=(tf.float32, tf.float32, tf.float32, tf.string))

# ==========================================================================
class NBCC_RGN(models.Model):

    def __init__(self):
        super(NBCC_RGN, self).__init__()

        # alphabetInit {'range':3.14159,'dist':'uniform'}
        self.alphabets = tf.random.uniform(shape=[alphabet_size, NUM_DIHEDRALS], minval=-3.14159, maxval=3.14159)

        # Bi-directional LSTM
        self.lstm1 = layers.Bidirectional(layers.LSTM(800, time_major=True, return_sequences=True, dropout=0.5), name='bi_lstm1')
        self.lstm2 = layers.Bidirectional(layers.LSTM(800, time_major=True, return_sequences=True, dropout=0.5), name='bi_lstm2')

        # dihedrals
        self.dense1 = layers.Dense(60, name='flatten')
        self.softmax = layers.Softmax(name='softmax')

    def call(self, inputs):
        # inputs : (N, batch_size, 62)
        num_steps = inputs.shape[0]

        x = self.lstm1(inputs)      # (N, batch_size, 1600)
        x = self.lstm2(x)           # (N, batch_size, 1600)

        # -------------------------------------------------
        # dihedrals
        x = self.dense1(x)          # (N, batch_size, 60)
        flat_x = tf.reshape(x, [-1, 60])    # (N x batch_size, 60)
        x = self.softmax(flat_x)            # (N x batch_size, 60)

        flat_dihedrals = reduce_mean_angle(x, self.alphabets)  # (N x batch_size, 60) * (60, 3) = (N x batch_size, 3)
        dihedrals = tf.reshape(flat_dihedrals, [num_steps, batch_size, NUM_DIHEDRALS])  # (N, batch_size, 3)

        # -------------------------------------------------
        # coordinates
        points = dihedral_to_point(dihedrals)       # (N x 3, batch_size, 3)
        coordinates = point_to_coordinate(points)   # (N x 3, batch_size, 3)

        return coordinates


# -------------------------------------------------
model = NBCC_RGN()
optimizer = optimizers.Adam(learning_rate=0.0001, beta_1=0.95, beta_2=0.99)

# model.fit() --> Please provide data which shares the same first dimension.

total_epochs = 10
current_epoch = 0



for epoch in range(total_epochs):
    # -------------------------------------------------
    # training
    current_step = 0
    _accumulate_loss = []

    for inputs, tertiaries, masks, ids in train_data:
        # print('inputs', inputs.shape)
        current_step += 1

        # tertiary masks
        _masks = []
        for i in range(masks.shape[0]):
            item_mask = tf.squeeze(tf.transpose(masks[i]), axis=0)
            mat_mask = masking_matrix(item_mask)
            _masks.append(mat_mask)

        ter_masks = tf.convert_to_tensor(_masks)
        ter_masks = tf.transpose(ter_masks, perm=(1, 2, 0), name='masks')
        weights, _ = _weights(ter_masks)

        with tf.GradientTape() as tape:
            pred_coord = model(inputs)

            # dRMSD Loss
            drmsds = _drmsds(pred_coord, tertiaries, weights)

            loss_value = tertiary_loss(drmsds, ter_masks)
            loss_value = tf.identity(loss_value)

            _accumulate_loss.append(float(loss_value))

        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # if current_step%100 == 0:
        #     print('{:03d} step : {}'.format(current_step, np.mean(_accumulate_loss)))

        if current_step == 1: break


    current_epoch += 1
    print('train {:03d} epoch : {} : {}'.format(current_epoch, len(_accumulate_loss), np.mean(_accumulate_loss)))


    # -------------------------------------------------
    # validation
    # for x, y in valid_data:

    # -------------------------------------------------
    # save checkpoint


# ==========================================================================
