import glob
import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(100)

phase = 'training'
file_list = sorted(glob.glob('../RGN11/data/ProteinNet11/{}/*'.format(phase)))

batch_size = 32

dataset = tf.data.TFRecordDataset(tf.data.Dataset.list_files(file_list)).batch(batch_size).shuffle(buffer_size=32)

def get_data():
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

    return ids, inputs, ter_y


def get_model(num_steps):
    model = keras.Sequential()

    # Bi-directional LSTM
    model.add(keras.layers.Bidirectional(
        keras.layers.LSTM(800, return_sequences=True, dropout=0.5), input_shape=(num_steps, 62), name='bi_lstm'
    ))
    model.add(keras.layers.Bidirectional(
        keras.layers.LSTM(800, return_sequences=True, dropout=0.5), name='bi_lstm_2'
    ))

    # TODO : --> dihedrals (alphabet) --> coordinates
    model.add(keras.layers.Dense(60, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(3))
    model.add(keras.layers.Softmax())

    model.compile(optimizer='adam', loss='mse', metrics=['mean_squared_error'])

    model.summary()

    return model

# --------------------------------------------------------------------------

model = get_model(batch_size)

# p_list = []
for k in range(300):
    id_list, X, y = get_data()
    print(X.shape, y.shape)

    hist = model.fit(X, y)

    # -------------------------------
    # 전체 데이터 사용여부 체크
    # m = id_list.numpy().tolist()
    # for item in m:
    #     p_list.append(item[0])
    #
    # if (k+1)%100 == 0:
    #     print(k+1, len(p_list), ':', len(set(p_list)))
    # -------------------------------



