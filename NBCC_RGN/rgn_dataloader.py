import glob
import tensorflow as tf

phase = 'testing'
file_list = sorted(glob.glob('../RGN11/data/ProteinNet11/{}/*'.format(phase)))

dataset = tf.data.TFRecordDataset(tf.data.Dataset.list_files(file_list)).batch(3)

for serialized_examples in dataset.take(1):

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
    print(ids)

    # int64 to int32
    primary = features['primary']
    primary = tf.dtypes.cast(primary, tf.int32)
    print(primary.shape)
    print(primary[0].shape)
    print(primary[1].shape)
    print(primary[2].shape)
    print()

    # convert to one_hot_encoding
    primary = tf.squeeze(primary, axis=2)
    print(primary.shape)
    one_hot_primary = tf.one_hot(primary, 20)
    print(one_hot_primary.shape)

    # padding
    one_hot_primary = tf.RaggedTensor.to_tensor(one_hot_primary)
    print(one_hot_primary.shape)

    # (batch_size, N, 20) to (N, batch_size, 20)
    one_hot_primary = tf.transpose(one_hot_primary, perm=(1, 0, 2))
    print(one_hot_primary.shape)
    print()

    print(one_hot_primary[40])
    print(one_hot_primary[100])
    print(one_hot_primary[200])






