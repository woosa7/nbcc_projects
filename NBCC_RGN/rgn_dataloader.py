import glob
import tensorflow as tf

NUM_DIHEDRALS = 3
NUM_DIMENSIONS = 3

batch_size = 3
tf.random.set_seed(100)

max_length = 700

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
        inputs, ter_y, masks = get_data(features)
        yield inputs, ter_y, masks, ids


train_data = tf.data.Dataset.from_generator(generator, output_types=(tf.float32, tf.float32, tf.float32, tf.string))
# train_data = tf.data.Dataset.from_generator(generator, output_types=(tf.float32, tf.float32))


# -----------------------------------------------------------------
def masking_matrix(mask, name=None):
    with tf.name_scope('masking_matrix') as scope:
        mask = tf.convert_to_tensor(mask, name='mask')

        mask = tf.expand_dims(mask, 0)
        base = tf.ones([tf.size(mask), tf.size(mask)])
        matrix_mask = base * mask * tf.transpose(mask)

        return matrix_mask

# -----------------------------------------------------------------
for x, y, masks, ids in train_data.take(1):
    print()
    print(x.shape)
    # print(y.shape)
    # print(ids.shape)

    print('masks', masks.shape)
    _masks = []
    for i in range(masks.shape[0]):
        a = tf.squeeze(tf.transpose(masks[i]), axis=0)
        c = masking_matrix(a)
        print(i, a.shape, c.shape)
        _masks.append(c)

    ter_masks = tf.convert_to_tensor(_masks)
    print('ter_masks', ter_masks.shape)
    ter_masks = tf.transpose(ter_masks, perm=(1, 2, 0), name='masks')
    print('ter_masks', ter_masks.shape)


# --------------------------------------------------------------------------------------------------------------------
