"""
utility functions of NBCC_RGN
"""

import numpy as np
import tensorflow as tf
from geom_ops import drmsd

# ==========================================================================
NUM_DIHEDRALS = 3
NUM_DIMENSIONS = 3

max_seq_length = 700

num_edge_residues = 0
loss_atoms = 'c_alpha'
tertiary_weight = 1.0
tertiary_normalization = 'first'
batch_dependent_normalization = True
num_evaluation_invocations = 1  # simple loss
LOSS_SCALING_FACTOR = 0.01      # this is to convert recorded losses to angstroms

# ==========================================================================
"""
functions from original rgn in model.py & net_ops.py
"""

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
        # mat = tf.sparse_to_dense(flat_indices, [max_seq_length, max_seq_length], flat_weights, validate_indices=False, name=scope)
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
        # Returns the batched diagonal part of a batched tensor.
        # (N, N, batch_size) --> (batch_size, N)
        traces = tf.linalg.diag_part(tf.transpose(masks, [2, 0, 1]))
        eff_stepss = tf.add(tf.reduce_sum(traces, [1]), num_edge_residues, name=scope) # NUM_EDGE_RESIDUES shouldn't be here, but I'm keeping it for
                                                                                       # legacy reasons. Just be clear that it's _always_ wrong to have
                                                                                       # it here, even when it's not equal to 0.
        return eff_stepss

@tf.function(experimental_relax_shapes=True)
def tertiary_loss(pred_coords, tertiaries, masks, weights, batch_size):
    """ Reduces loss according to normalization order.
        this = _reduce_loss_quotient() + _accumulate_loss() in original RGN

        * training mode
        num_evaluation_invocations = 1
        filter - all

        * evaluation mode
        num_evaluation_invocations = 7
        subgroups based on maximum sequence identity to the training set
        filter - all, 10, 20, 30, 40, 50, 70, 90
    """
    drmsds = _drmsds(pred_coords, tertiaries, weights)   # (batch_size,)

    # NBCC modified for tf 2.0
    filters = {'all': tf.tile([True], [batch_size,])}  # training mode
    group_filter = filters['all']

    # losses_filtered == drmsds in filters['all']
    losses_filtered = tf.boolean_mask(drmsds, group_filter) # it will give problematic results if all entries are removed

    # tertiary_normalization == 'first':
    # effective number of residues that are non-missing
    loss_factors = tf.boolean_mask(effective_steps(masks, num_edge_residues), group_filter)

    # *** original code
    # if tertiary_normalization == 'zeroth':
    #     loss_factors = tf.ones_like(losses_filtered)
    # elif tertiary_normalization == 'first':
    #     loss_factors = tf.boolean_mask(effective_steps(masks, num_edge_residues), group_filter)
    #     fixed_denominator_factor = float(max_seq_length - num_edge_residues)
    # elif tertiary_normalization == 'second':
    #     eff_num_stepss = tf.boolean_mask(effective_steps(masks, num_edge_residues), group_filter)
    #     loss_factors = (tf.square(eff_num_stepss) - eff_num_stepss) / 2.0
    #     fixed_denominator_factor = float(max_seq_length - num_edge_residues)
    #     fixed_denominator_factor = ((fixed_denominator_factor ** 2) - fixed_denominator_factor) / 2.0

    # batch_dependent_normalization = True
    numerator   = tf.reduce_sum(loss_factors * losses_filtered)
    denominator = tf.reduce_sum(loss_factors)

    # *** original code
    # if batch_dependent_normalization or tertiary_normalization == 'zeroth':
    #     denominator = tf.reduce_sum(loss_factors)
    # else:
    #     denominator = tf.multiply(tf.cast(tf.size(loss_factors), tf.float32), fixed_denominator_factor)

    # NBCC modified for tf 2.0
    # _accumulate_loss() --> simple loss
    tertiary_loss = tf.math.divide(numerator, denominator) * tertiary_weight * LOSS_SCALING_FACTOR

    return tertiary_loss, losses_filtered, loss_factors, drmsds * LOSS_SCALING_FACTOR

def accumulated_loss(losses_filtered, loss_factors):
    """
    Accumulate losses for all batches of one epoch.
    """
    losses_filtered = tf.convert_to_tensor(losses_filtered)
    loss_factors = tf.convert_to_tensor(loss_factors)

    numerator = tf.reduce_sum(loss_factors * losses_filtered) # = effective_steps * drmsds
    denominator = tf.reduce_sum(loss_factors)

    return tf.math.divide(numerator, denominator) * tertiary_weight * LOSS_SCALING_FACTOR

def mask_and_weight(masks):
    # tertiary masks
    _masks = []
    for i in range(masks.shape[0]):
        item_mask = tf.squeeze(tf.transpose(masks[i]), axis=0)
        mat_mask = masking_matrix(item_mask)
        _masks.append(mat_mask)

    ter_masks = tf.convert_to_tensor(_masks)
    ter_masks = tf.transpose(ter_masks, perm=(1, 2, 0), name='masks')

    # weights
    weights, _ = _weights(ter_masks)

    return ter_masks, weights


# ==========================================================================
"""
Data Loader
* less 700 residues
training dataset    : 41789 / 32 = 1305
validation dataset  :   224 / 32 =    7
test dataset        :    81 / 32 =    2
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
    primary = tf.squeeze(primary, axis=2)  # (32, None, None) --> (32, None)
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

    return inputs, tertiary, mask, primary

# ==========================================================================
