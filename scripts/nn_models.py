# =========================================================================
# Imports
# =========================================================================

import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.python.keras.layers.core import Flatten, RepeatVector


# 'input_length' and 'output_length' are set to 9 & 8, respectively, as these
# match the chem1 network (8 species + temperature input, 8 species output)

# =========================================================================
# Custom Layers (Classes)
# =========================================================================
class CustomAttention(keras.layers.Layer):
  def __init__(self):
    super(CustomAttention, self).__init__()

  def build(self, input_shape):
    self.W = self.add_weight(name="att_weight", shape=(
        input_shape[-1], 1), initializer='normal')
    self.b = self.add_weight(name="att_bias", shape=(
        input_shape[1], 1), initializer='zeros')

    super(CustomAttention, self).build(input_shape)

  def call(self, x):
    e = K.tanh(K.dot(x, self.W) + self.b)
    a = K.softmax(e, axis=1)

    output = x * a

    return output

# =========================================================================
# Pre-defined models
# =========================================================================


def simple_rnn(input_length=9, num_features=1, output_length=1):
  print("Creating simple RNN")
  model = keras.models.Sequential([
      keras.layers.SimpleRNN(20, input_shape=[input_length, num_features]),
      keras.layers.RepeatVector(output_length),
      keras.layers.TimeDistributed(
          keras.layers.Dense(num_features, activation="linear")
      ),
  ])

  return model


def deep_rnn(input_length=9, num_features=1, output_length=1):
  print("Creating deep RNN")
  model = keras.models.Sequential([
      keras.layers.SimpleRNN(20, return_sequences=False,
                             input_shape=[input_length, num_features]),
      keras.layers.RepeatVector(output_length),
      keras.layers.SimpleRNN(20, return_sequences=True),
      keras.layers.TimeDistributed(
          keras.layers.Dense(num_features, activation="linear")
      ),
  ])

  return model


def lstm_attention_model(input_length=9, num_features=1, output_length=1):
  # Similar to 'lstm_model' but builds an additional attention network
  print("Creating attention LSTM")
  model = keras.models.Sequential([
      keras.layers.LSTM(32, input_shape=(input_length, num_features),
                        return_sequences=True),
      CustomAttention(),
      keras.layers.LSTM(32, return_sequences=True),
      keras.layers.TimeDistributed(
          keras.layers.Dense(num_features, activation='linear')
      )
  ])

  return model


def conv_gru_model(input_length=9, num_features=1, output_length=1):
  # 1D conv layer to process sequences, then GRU cells for time series
  # predictions
  model = keras.models.Sequential([
      keras.layers.Conv1D(filters=20, kernel_size=4, strides=2,
                          padding="valid",
                          input_shape=[input_length, num_features]),
      keras.layers.GRU(20, return_sequences=False),
      keras.layers.RepeatVector(output_length),
      keras.layers.GRU(20, return_sequences=True),
      keras.layers.TimeDistributed(
          keras.layers.Dense(num_features, activation="linear")
      ),
  ])

  return model


def cnn_1d_model(input_length=9, num_features=1, output_length=8,
                 num_filters=4, kernel_sizes=[3, 5], pool_size=2):
  print("Creating 1D CNN")
  model = keras.models.Sequential([
      keras.layers.Conv1D(num_filters, kernel_sizes[0],
                          input_shape=(input_length, num_features),
                          activation='relu', padding='same'),
      keras.layers.Conv1D(num_filters, kernel_sizes[1],
                          activation='relu'),
      keras.layers.MaxPooling1D(pool_size),
      keras.layers.Flatten(),
      keras.layers.Dense(output_length * num_features, activation='relu'),
      keras.layers.Reshape((output_length, num_features)),
  ])

  return model


def simple_dnn(input_length=9, output_length=8):
  model = keras.models.Sequential([
      # Input
      keras.layers.Dense(
          output_length*4, input_shape=(input_length,), activation="sigmoid"),
      # Dense Layer(s)
      keras.layers.Dense(output_length*2, activation="sigmoid"),
      # Output
      keras.layers.Dense(output_length*2, activation="relu"),
  ])

  return model


# =========================================================================
# Customisable Models (to use with optuna)
# =========================================================================
def stacked_lstm(input_length=9, output_length=1, num_features=1,
                 num_layers=1, layers=[8],
                 use_attention=False):
  model = keras.models.Sequential()
  layer = keras.layers.LSTM  # could later add other layers to be chosen

  if num_layers == 1:  # single-layer LSTM
    model.add(layer(layers[0],
                    input_shape=(input_length, num_features),
                    return_sequences=use_attention))
  else:  # multi-layer LSTM
    for i in range(num_layers):
      if i == 0:  # first layer
        if i + 1 != num_layers:
          model.add(layer(layers[i],
                          input_shape=(input_length, num_features),
                          return_sequences=use_attention))
        else:
          model.add(layer(layers[i], input_shape=(
              input_length, num_features)))
      elif i + 1 == num_layers:  # last layer
        model.add(layer(layers[i], return_sequences=True))
      else:  # intermediate layers
        model.add(layer(layers[i], return_sequences=use_attention))

  # Add possibility of multiple Dense layers!
  model.add(keras.layers.Dense(num_features, 'linear'))

  return model


def encoder_decoder_lstm(input_length=9, output_length=1, num_features=1,
                         num_encoder_layers=1, encoder_layers=[32],
                         num_decoder_layers=1, decoder_layers=[32],
                         use_attention=False):
  model = keras.models.Sequential()
  layer = keras.layers.LSTM  # could later add other layers to be chosen

  # Encoder
  # Attention needs 'return_sequences=True', repeat vector needs 'False'
  if num_encoder_layers == 1:  # single layer encoder
    model.add(layer(encoder_layers[0],
                    input_shape=(input_length, num_features),
                    return_sequences=use_attention))
  else:
    for i in range(num_encoder_layers):
      if i == 0:  # First layer in multi-layer encoder
        model.add(layer(encoder_layers[i],
                        input_shape=(input_length, num_features),
                        return_sequences=True))
      elif i + 1 == num_encoder_layers:  # Last layer
        model.add(layer(encoder_layers[i],
                        return_sequences=use_attention))
      else:  # intermediate layers
        model.add(layer(encoder_layers[i], return_sequences=True))

  # Attention or Repeat Vector
  if use_attention:
    model.add(CustomAttention())
  else:
    model.add(RepeatVector(output_length))

  # Decoder
  for i in range(num_decoder_layers):
    layer = keras.layers.LSTM
    model.add(layer(decoder_layers[i], return_sequences=True))

  # Time Distributed layer
  model.add(keras.layers.TimeDistributed(
      keras.layers.Dense(num_features, activation='linear')
  ))

  return model


def stacked_cnn_1d(input_length=9, output_length=8, num_features=1,
                   num_conv_blocks=2, layers_per_block=[1, 2],
                   filters=[4, 4], kernel_sizes=[3, 5],
                   pool_sizes=[2, 3],
                   num_dense_layers=0, dense_neurons=[]):
  # Stacked 1D CNN model to be used with optuna
  # Organised into CNN blocks of layers with MaxPooling in between and
  # Dense at the end
  # By default there are no extra Dense layers
  model = keras.models.Sequential()
  layer = keras.layers.Conv1D

  # CNN Blocks
  for i in range(num_conv_blocks):  # Each block consists of 'n' CNN layers
    num_conv_layers = layers_per_block[i]
    for j in range(num_conv_layers):
      if j == 0:  # First block (want to check for first layer of network)
        _filter = filters[i]
        if i == 0:  # First layer of network
          conv_layer = layer(_filter, kernel_sizes[i], activation='relu',
                             input_shape=(input_length, num_features))
        else:
          conv_layer = layer(_filter, kernel_sizes[i], activation='relu')
      else:
        # Filter is 50% larger in each subsequent layer
        _filter = int(_filter * 1.5)
        conv_layer = layer(_filter, kernel_sizes[i], activation='relu')

      model.add(conv_layer)

    # Add MaxPooling after each block
    # Should also add BatchNormalisation
    model.add(keras.layers.MaxPooling1D(pool_sizes[i]))

  # Flatten Conv layers to feed into Dense
  model.add(keras.layers.Flatten())

  # Dense Layers
  # Should also add Dropout
  layer = keras.layers.Dense
  for i in range(num_dense_layers):
    model.add(layer(dense_neurons[i]))

  # Output; final Dense layer and a Reshape to ensure output shape
  model.add(layer(output_length * num_features, activation='relu'))
  model.add(keras.layers.Reshape((output_length, num_features)))

  return model
