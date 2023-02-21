# Prototype a ResNet
from sklearn.ensemble import RandomForestRegressor
import tensorflow.keras as keras
import numpy as np
from nn_models import simple_dnn
from abundances import *
from typing import Dict, List
import glob
import vaex
import tensorflow as tf
import tensorflow_gnn as tfgnn
from petastorm import make_batch_reader
from petastorm.tf_utils import make_petastorm_dataset
from gcrn.helper_functions import number_densities_from_abundances
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import keras_tuner as kt
import xgboost as xgb
import pickle


# Input: abundances, gas density, temperature (10 quantities for chem1)
# Output: equilibrium number densities (8 quantities for chem1)
"""
TODO
  - Write architectures for
    - ResNet
  - Check normal ML algs
    - SVMs
    - Decision Trees/Random Forests
    - XGBoost
  - Functions to standardise input
"""

"""
Pipeline overview:
  - Read in .parquet files
  - Generate dataset (extract density, temperature, EQ number densities)
  - Determine and calculate abundances based on keys
  - Sort input abundance array and output EQ array alphabetically
  - Pass into model with goal:
    - from density, temperature and abundance, map to output EQ number densities
  - Loss function analysis
"""


class WeightedSumConvolution(tf.keras.layers.Layer):
  "Weighted sum of source node states"

  def __call__(self, graph: tfgnn.GraphTensor,
               edge_set_name: tfgnn.EdgeSetName) -> tfgnn.Field:
    messages = tfgnn.broadcast_node_to_edges(
        graph, edge_set_name, tfgnn.SOURCE, feature_name=tfgnn.DEFAULT_STATE_NAME
    )
    weights = graph.edge_sets[edge_set_name]["weight"]
    weighted_messages = tf.expand_dims(weights, -1) * messages
    pooled_messages = tfgnn.pool_edges_to_node(
        graph, edge_set_name, tfgnn.TARGET, reduce_type="sum",
        feature_value=weighted_messages
    )

    return pooled_messages


def gnn_network():
  gnn = tfgnn.keras.ConvGNNBuilder(
      lambda edge_set_name: WeightedSumConvolution(),
      lambda node_set_name: tfgnn.keras.layers.NextStateFromConcat(
          tf.keras.layers.Dense(32)
      )
  )
  model = keras.Sequential([
      gnn.Convolve()
  ])

  return model


def mlp_builder(hp):
  # Builder function for keras tuner
  model = keras.Sequential()
  hp_units = hp.Int(min_value=16, max_value=256, step=32)
  # Input
  model.add(keras.layers.Dense(32, input_shape=(6,), activation="relu"))

  # Layers to tune
  model.add(keras.layers.Dense(units=hp.units, activation="relu"))
  keras.layers.Dropout(0.5)
  model.add(keras.layers.Dense(units=hp.units, activation="relu"))
  keras.layers.Dropout(0.5)
  model.add(keras.layers.Dense(units=hp.units, activation="relu"))
  keras.layers.Dropout(0.5)

  # Output
  model.add(keras.layers.Dense(8, activation="linear"))

  # Learning rate
  hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
  model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss="mse", metrics=["mae"])

  return model


def mlp():
  model = keras.Sequential([
      keras.layers.Dense(32, input_shape=(6,), activation="relu"),
      keras.layers.Dense(32, activation="relu"),
      keras.layers.Dense(32, activation="relu"),
      keras.layers.Dense(8, activation="linear"),
  ])

  model.compile(optimizer="adam", loss="mse", metrics=["mae"])
  return model


def mlp_dropout():
  # Same as mlp() but with Dropout layers between each Dense layer
  model = keras.Sequential([
      keras.layers.Dense(64, input_shape=(6,), activation="relu"),
      keras.layers.Dropout(0.5),
      keras.layers.Dense(64, activation="relu"),
      keras.layers.Dropout(0.5),
      keras.layers.Dense(64, activation="relu"),
      keras.layers.Dropout(0.5),
      keras.layers.Dense(8, activation="linear"),
  ])

  model.compile(optimizer="adam", loss="mse", metrics=["mae"])
  return model


def cnn_1d():
  # 1D CNN
  model = keras.Sequential([
      # # Input
      # keras.layers.Dense(32, input_shape=(6,), activation="relu"),
      # Convolutional Layers
      keras.layers.Conv1D(32, 4, input_shape=(
          6, 1), activation="relu", padding="same"),
      keras.layers.Conv1D(64, 4, activation="relu"),
      keras.layers.MaxPooling1D(2),
      keras.layers.Flatten(),
      # Output
      keras.layers.Dense(8, activation="linear")
  ])

  model.compile(optimizer="adam", loss="mse", metrics=["mae"])
  return model


def encoder_decoder():
  model = keras.Sequential([
      # Encoder
      keras.layers.Dense(128, input_shape=(6,), activation="relu"),
      keras.layers.Dense(64, activation="relu"),
      keras.layers.Dense(32, activation="relu"),
      # Decoder
      keras.layers.Dense(32, activation="relu"),
      keras.layers.Dense(64, activation="relu"),
      keras.layers.Dense(128, activation="relu"),
      # Output
      keras.layers.Dense(8, activation="linear")
  ])

  model.compile(optimizer="adam", loss="mse", metrics=["mae"])
  return model


def random_forest():
  model = RandomForestRegressor(n_estimators=100, criterion="mae", verbose=1)
  return model


def load_dataset(directory: str, suffix="*.parquet"):
  files = [f"file://{f}" for f in glob.glob(f"{directory}/{suffix}")]
  print(files)
  with make_batch_reader(files) as reader:
    dataset = make_petastorm_dataset(reader)
    iterator = dataset.make_one_shot_iterator()
    tensor = iterator.get_next()
    with tf.Session() as sess:
      sample = sess.run(tensor)
      print(sample.id)


def load_dataframe(path: str):
  return vaex.open(path)


def minmax(arr):
  return np.nanmin(arr), np.nanmax(arr)


def print_stats(stats):
  for model_name, losses in stats.items():
    print(f"{model_name}: {losses}")


def add_log_n_cols(df, abundances, species):
  mass_hydrogen = 1.67262171e-24  # g
  abundance_arr = np.array([v for v in abundances.values()])
  percent_hydrogen = 10**abundances['H'] / (np.sum(10**abundance_arr))
  df["log10_H_density"] = np.log10(
      df.density / (percent_hydrogen * mass_hydrogen))
  for s in species:
    print(f"Adding {abundances[s]} to column N_{s}")
    df[f"N_{s}"] = df.log10_H_density + abundances[s] - 12

  return df


def add_abundance_cols(df, abundances, species):
  for s in species:
    print(f"Adding {abundances[s]} to column A_{s}")
    df[f"A_{s}"] = vaex.vconstant(abundances[s], len(df))

  return df


def prepare_dataset():
  cemp_dir = "/media/sdeshmukh/Crucial X6/mean_chemistry/combined_cemp"
  suffix = "*chem1*.txt"

  # species = ["H", "H2", "C", "O", "CO", "CH", "OH", "M"]
  species = ["H", "C", "O", "M"]
  chem_keys = [f"{s}_EQ" for s in ["H", "H2", "C", "O", "CO", "CH", "OH", "M"]]
  input_keys = [*[f"A_{s}" for s in species], "density", "temperature"]
  # input_keys = [*[f"N_{s}" for s in species], "density", "temperature"]
  print("Input")
  print(input_keys)
  print("Output")
  print(chem_keys)
  # test_suffix = "d3t63g40mm30chem1_04*.parquet"  # load one file
  # TODO:
  # - Use 5 files with different abundances and correctly assign the required
  #   abundance to each
  snap_num = "101"
  test_files = [f"{cemp_dir}/d3t63g40mm{id_}chem1_{snap_num}.parquet" for id_ in
                ["20", "30", "30c20n20o20", "30c20n20o04"]]
  abundance_mappings = [mm20a04_abundances, mm30a04_abundances,
                        mm30a04c20n20o20_abundances, mm30a04c20n20o04_abundances]
  dfs = [load_dataframe(f) for f in test_files]
  for df_, abundances in zip(dfs, abundance_mappings):
    df_ = add_abundance_cols(df_, abundances, species)
    # df_ = add_log_n_cols(df_, abundances, species)
    # print(list(df_.columns))

  df = vaex.concat([*dfs])

  # TODO: Combine dfs into 1?
  # Single file
  # test_file = "d3t63g40mm30chem1_08*.parquet"
  # df = load_dataframe(f"{cemp_dir}/{test_file}")
  # print("Loaded DF")

  # # Let's just pass the abundances in
  # abundances = np.array([mm30a04_abundances[s] for s in species])
  # new_cols = [f"A_{s}" for s in species]

  # n_elements = df['density'].shape[0]
  # for i, k in enumerate(new_cols):
  #   df[k] = np.repeat(abundances[i], n_elements)
  # df = add_log_n_cols(df, mm30a04_abundances, species)
  # df = add_abundance_cols(df, mm30a04_abundances, species)

  data = df[input_keys]
  print("Added new columns")
  print(list(data.columns))
  data["density"] = np.log10(data["density"])
  # data["temperature"] = np.log10(data["temperature"])
  label = df[chem_keys]
  for k in chem_keys:
    label[k] = np.log10(label[k])
  print("Applied log scaling")

  data_rest, data_test, label_rest, label_test = train_test_split(
      data.values, label.values, test_size=0.25, random_state=42)
  data_train, data_val, label_train, label_val = train_test_split(
      data_rest, label_rest, test_size=0.1, random_state=42)
  print("Split data")
  print("Train", data_train.shape, label_train.shape)
  print("Val", data_val.shape, label_val.shape)
  print("Test", data_test.shape, label_test.shape)
  print(data_train[0])
  print(minmax(data_train[:, 0]))
  scaler = MinMaxScaler(feature_range=(0, 1))
  scaler.fit(data_train)
  data_train = scaler.transform(data_train)
  data_val = scaler.transform(data_val)
  data_test = scaler.transform(data_test)

  print(data_train[0])
  print("Any nans?")
  for k in chem_keys:
    print(k, np.any(np.isnan(label[k].values)))

  return (data_train, label_train), (data_val, label_val), (data_test, label_test)


def prepare_models(uid=""):
  # Set up sklearn, XGBoost and TensorFlow models
  models = [
      # (f"SKL_Linear{uid}", LinearRegression(n_jobs=4)),
      # (f"SKL_DT{uid}", DecisionTreeRegressor(criterion="mse")),
      # (f"SKL_RF{uid}", RandomForestRegressor(
      #     criterion="mse", n_jobs=2, verbose=1)),
      # (f"XGB_XGB{uid}", xgb.XGBRegressor()),
      (f"TF_MLP{uid}", mlp()),
      (f"TF_MLPD{uid}", mlp_dropout()),
      (f"TF_CNN{uid}", cnn_1d()),
      (f"TF_EncDec{uid}", encoder_decoder),
  ]

  return models


def train_models(models, X_train, y_train, X_val, y_val, X_test, y_test):
  # Trains models from SKLearn, XGBoost and TensorFlow
  model_stats = {}
  for model_name, model in models:
    print(f"Training \'{model_name}\'")
    if model_name.startswith("TF_"):
      model.summary()
      # TensorFlow training
      checkpoint_path = f"./models/{model_name}.ckpt"
      checkpoint_cb = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                      save_weights_only=True,
                                                      verbose=1)
      history = model.fit(X_train, y_train, batch_size=512, epochs=500,
                          validation_data=(X_val, y_val),
                          callbacks=[keras.callbacks.EarlyStopping(restore_best_weights=True,
                                                                   patience=30),
                                     checkpoint_cb,
                                     keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                                       factor=0.1, patience=30, min_delta=1e-4, cooldown=0,
                                                                       min_lr=1e-4)],
                          verbose=2)
      with open(f"./train_history/{model_name}.pkl", "wb") as history_file:
        pickle.dump(history.history, history_file)

      model_stats[model_name] = model.evaluate(X_test, y_test)

    else:
      # SKLearn/XGBoost training
      model.fit(X_train, y_train)
      # Evaluate
      y_pred = model.predict(X_test)
      mse = mean_squared_error(y_test, y_pred)
      mae = mean_absolute_error(y_test, y_pred)
      model_stats[model_name] = (mse, mae)
      save_path = f"./models/{model_name}.pkl"
      pickle.dump(model, open(save_path, "wb"))

    chem_keys = [f"{s}_EQ" for s in [
        "H", "H2", "C", "O", "CO", "CH", "OH", "M"]]
    predictions, truths, inputs = check_random_idx(model, X_test, y_test,
                                                   seed=None)
    print(model_name)
    print(model_stats[model_name])
    for j in range(5):
      for i, s in enumerate(chem_keys):
        print(
            f"({s}): Prediction = {predictions[j, i]:.3f}. Truth = {truths[j, i]:.3f}. P - T = {predictions[j, i] - truths[j, i]:.3f}")
        print(f"Inputs: {inputs[j]}")

  return models, model_stats


def main():
  # TensorFlow models
  uid = "6-10"  # maps X features - Y outputs
  models = prepare_models(uid=uid)
  print("Set up models")
  (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_dataset()
  print("Set up dataset")
  models, model_stats = train_models(models, X_train, y_train, X_val, y_val,
                                     X_test, y_test)
  print_stats(model_stats)


def check_random_idx(model, X_test, y_test, seed=42):
  if seed:
    np.random.seed(seed)
  idx = np.random.randint(0, X_test.shape[0])
  predictions = model.predict(X_test[idx:idx+10])
  truths = y_test[idx:idx+10]

  return predictions, truths, X_test[idx:idx+10]


if __name__ == "__main__":
  main()
