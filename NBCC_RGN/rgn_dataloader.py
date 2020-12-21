import glob
import tensorflow as tf

NUM_DIHEDRALS = 3
NUM_DIMENSIONS = 3

batch_size = 32
tf.random.set_seed(100)

# training dataset
# all       : 42528 = 32 * 1329
# less 700  : 41789 = 32 * 1306

phase = 'training'
file_list = glob.glob('./RGN11/data/ProteinNet11/{}/*'.format(phase))

# --------------------------------------------------------------------------
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

    return ids, features

def filter_func(ids, features):
    primary = features['primary']
    primary = tf.dtypes.cast(primary, tf.int32)

    pri_length = tf.size(primary)
    keep = pri_length <= 700

    return keep


dataset = tf.data.TFRecordDataset(tf.data.Dataset.list_files(file_list, shuffle=True)).map(mapping_func).filter(filter_func).batch(batch_size)


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



p_list = []
# for element in dataset.take(10000):
for k, element in enumerate(dataset.repeat()):
    ids = element[0]
    m = ids.numpy().tolist()
    for item in m:
        p_list.append(item[0])
    if (k+1)%100 == 0:
        print('----------------', k+1, len(p_list), ':', len(set(p_list)))

    features = element[1]
    inputs, ter_y = get_data(features)

    if inputs.shape[0] >= 700:
        print(k, inputs.shape)
