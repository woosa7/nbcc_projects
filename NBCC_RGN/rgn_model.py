from tensorflow import keras

def get_lstm_model(num_steps):
    model = keras.Sequential()
    model.add(
        # forward
        keras.layers.LSTM(800, input_shape=(num_steps, 62), return_sequences=True, dropout=0.5, name='lstm_1')
    )
    model.add(
        # backward : input sequence in reverse
        keras.layers.LSTM(800, return_sequences=True, dropout=0.5, go_backwards=True, name='lstm_2')
    )
    model.add(
        # forward
        keras.layers.LSTM(800, return_sequences=True, dropout=0.5, name='lstm_3'),
    )
    model.add(
        # backward : input sequence in reverse
        keras.layers.LSTM(800, return_sequences=True, dropout=0.5, go_backwards=True, name='lstm_4')
    )

    model.summary()
    return model

def get_bi_lstm_model(num_steps):
    model = keras.Sequential()
    model.add(
        keras.layers.Bidirectional(
            keras.layers.LSTM(800, return_sequences=True, dropout=0.5), input_shape=(num_steps, 62), name='bi_lstm'
        )
    )
    model.summary()

    return model

# -------------------------------------------------------------
batch_size = 32
num_steps = batch_size

model1 = get_lstm_model(num_steps)

model2 = get_bi_lstm_model(num_steps)
