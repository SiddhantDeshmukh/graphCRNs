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


class Config:
  def __init__(self, num_inputs: int, num_outputs: int, uid_suffix="",
               input_species=[], output_species=[], use_logn=False) -> None:
    # input_keys & output_keys must have len (num_inputs - 2) & num_outputs, respectively
    self.num_inputs = num_inputs
    self.num_outputs = num_outputs
    uid_logn_id = "logn" if use_logn else "abu"
    self.uid = f"{num_inputs}-{num_outputs}_{uid_logn_id}{uid_suffix}"
    self.input_keys = input_species
    self.output_keys = output_species
    # whether to use log(n) inputs (True) or abundance inputs (False)
    self.use_logn = use_logn


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


def mlp(input_shape=(6,), num_out=8):
  model = keras.Sequential([
      keras.layers.Dense(32,  input_shape=input_shape, activation="relu"),
      keras.layers.Dense(32, activation="relu"),
      keras.layers.Dense(32, activation="relu"),
      keras.layers.Dense(num_out, activation="linear"),
  ])

  model.compile(optimizer="adam", loss="mse", metrics=["mae"])
  return model


def mlp_dropout(input_shape=(6,), num_out=8):
  # Same as mlp() but with Dropout layers between each Dense layer
  model = keras.Sequential([
      keras.layers.Dense(64, input_shape=input_shape, activation="relu"),
      keras.layers.Dropout(0.5),
      keras.layers.Dense(64, activation="relu"),
      keras.layers.Dropout(0.5),
      keras.layers.Dense(64, activation="relu"),
      keras.layers.Dropout(0.5),
      keras.layers.Dense(num_out, activation="linear"),
  ])

  model.compile(optimizer="adam", loss="mse", metrics=["mae"])
  return model


def cnn_1d(input_shape=(6, 1), num_out=8):
  # 1D CNN
  model = keras.Sequential([
      # # Input
      # keras.layers.Dense(32, input_shape=(6,), activation="relu"),
      # Convolutional Layers
      keras.layers.Conv1D(32, 4, input_shape=input_shape,
                          activation="relu", padding="same"),
      keras.layers.Conv1D(64, 4, activation="relu"),
      keras.layers.MaxPooling1D(2),
      keras.layers.Flatten(),
      # Output
      keras.layers.Dense(num_out, activation="linear")
  ])

  model.compile(optimizer="adam", loss="mse", metrics=["mae"])
  return model


def encoder_decoder(input_shape=(6,), num_out=8):
  model = keras.Sequential([
      # Encoder
      keras.layers.Dense(128, input_shape=input_shape, activation="relu"),
      keras.layers.Dense(64, activation="relu"),
      keras.layers.Dense(32, activation="relu"),
      # Decoder
      keras.layers.Dense(32, activation="relu"),
      keras.layers.Dense(64, activation="relu"),
      keras.layers.Dense(128, activation="relu"),
      # Output
      keras.layers.Dense(num_out, activation="linear")
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
  log10_H_density = np.log10(df.density / (percent_hydrogen * mass_hydrogen))
  for s in species:
    print(f"Adding {abundances[s]} to column N_{s}")
    # Apparently vaex doesn't recognise these as columns but it still works?
    df[f"N_{s}"] = log10_H_density + abundances[s] - 12

  return df


def add_abundance_cols(df, abundances, species):
  for s in species:
    print(f"Adding {abundances[s]} to column A_{s}")
    df[f"A_{s}"] = vaex.vconstant(abundances[s], len(df), dtype=np.float32)

  return df


def clean_dfs(*dfs):
  # Remove all unnecessary columns
  drop_cols = ["density_centres", "temperature_centres", "x", "y", "z", "abs_v"]
  for df in dfs:
    for col in df.columns:
      if col in drop_cols or col.endswith("_NEQ"):
        print(f"Dropping {col}")
        df = df.drop(col, inplace=True)
      # else:
      #   df[col] = df[col].astype("float32")

  return dfs


def check_dfs(*dfs):
  for i, df in enumerate(dfs):
    print(f"DF {i+1}: {len(list(df.columns))} cols, {len(df)} rows")
    print("dtypes:")
    print(df.dtypes)
    print("shapes: ", df.shape)
    # print("Any Missing?", df.ismissing().any())
    # print("Any NaN?", df.isnan().any())
    # print("Any inf?", df.isinfinite().any())


def prepare_dataset(use_logn=False,
                    input_species=["H", "H2", "C", "O", "CO", "CH", "OH", "M"],
                    output_species=["H", "H2", "C", "O", "CO", "CH", "OH", "M"],
                    test_files=None, abundance_mappings=None):
  cemp_dir = "/media/sdeshmukh/Crucial X6/mean_chemistry/combined_cemp"

  if use_logn:
    X_keys = [*[f"N_{s}" for s in input_species], "density", "temperature"]
  else:
    X_keys = [*[f"A_{s}" for s in input_species], "density", "temperature"]
  y_keys = [f"{s}_EQ" for s in output_species]

  # test_suffix = "d3t63g40mm30chem1_04*.parquet"  # load one file
  # TODO:
  # - Use 5 files with different abundances and correctly assign the required
  #   abundance to each
  # snap_num = "017"
  snap_num = "07*"
  # snap_num = "023"
  if test_files is None or abundance_mappings is None:
    test_files = [f"{cemp_dir}/d3t63g40mm{id_}chem1_{snap_num}.parquet" for id_ in
                  [
                      # "00",
                      # "20",
                      # "30",
                      "30c20n20o20",
                      # "30c20n20o04"
                  ]]
    abundance_mappings = [
        # mm00_abundances,
        # mm20a04_abundances,
        # mm30a04_abundances,
        mm30a04c20n20o20_abundances,
        # mm30a04c20n20o04_abundances
    ]
  dfs = [load_dataframe(f) for f in test_files]
  dfs = clean_dfs(*dfs)
  for df_, abundances in zip(dfs, abundance_mappings):
    if use_logn:
      df_ = add_log_n_cols(df_, abundances, input_species)
    else:
      df_ = add_abundance_cols(df_, abundances, input_species)

  print(list(dfs[0].columns))
  print("Cleaned DFs")
  check_dfs(*dfs)
  # exit()
  print("Concatenating DataFrames")
  df = vaex.concat([*dfs], resolver="strict")
  print(list(df.columns))

  print("Input")
  print(X_keys)
  print("Output")
  print(y_keys)
  print(list(df.columns))

  data = df[X_keys]
  label = df[y_keys]
  print("Input columns:")
  print(list(data.columns))
  print("Output columns:")
  print(list(label.columns))

  # Apply log scaling
  data["density"] = np.log10(data["density"])
  # data["temperature"] = np.log10(data["temperature"])
  for k in y_keys:
    label[k] = np.log10(label[k])
  print("Applied log scaling")

  # .values throws TypeError: DataType expected, got <class 'numpy.dtype[float32]'>
  # when using multiple DFs (concatenating seems to mess up dtypes?)
  X = data.values
  y = label.values
  X_trainval, X_test, y_trainval, y_test = train_test_split(
      X, y, test_size=0.25, random_state=42)
  X_train, X_val, y_train, y_val = train_test_split(
      X_trainval, y_trainval, test_size=0.1, random_state=42)
  print("Split data")
  print("Train", X_train.shape, y_train.shape)
  print("Val", X_val.shape, y_val.shape)
  print("Test", X_test.shape, y_test.shape)
  print(X_train[0])
  for i in range(X_train.shape[1]):
    print(X_keys[i], minmax(X_train[:, i]))
  scaler = MinMaxScaler(feature_range=(0, 1))
  scaler.fit(X_train)
  X_train = scaler.transform(X_train)
  X_val = scaler.transform(X_val)
  X_test = scaler.transform(X_test)

  print(X_train[0])
  print("Any nans?")
  for k in y_keys:
    print(k, np.any(np.isnan(label[k].values)))

  return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def prepare_models(num_in: int, num_out: int, uid=""):
  input_shape = (num_in,)
  # Set up sklearn, XGBoost and TensorFlow models
  models = [
      # (f"SKL_Linear{uid}", LinearRegression(n_jobs=4)),
      # (f"SKL_DT{uid}", DecisionTreeRegressor(criterion="mse")),
      # (f"SKL_RF{uid}", RandomForestRegressor(
      #     criterion="mse", n_jobs=2, verbose=1)),
      # (f"XGB_XGB{uid}", xgb.XGBRegressor()),
      (f"TF_MLP{uid}", mlp(input_shape=input_shape, num_out=num_out)),
      # (f"TF_MLPD{uid}", mlp_dropout(input_shape=input_shape, num_out=num_out)),
      (f"TF_CNN{uid}", cnn_1d(input_shape=(num_in, 1), num_out=num_out)),
      (f"TF_EncDec{uid}", encoder_decoder(input_shape=input_shape,
                                          num_out=num_out)),
  ]

  return models


def train_models(models, X_train, y_train, X_val, y_val, X_test, y_test, outputs):
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
      history = model.fit(X_train, y_train, batch_size=512,
                          epochs=500,
                          # epochs=5,
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

      model_stats[model_name] = model.evaluate(X_test, y_test, verbose=2)

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

    predictions, truths, inputs = check_random_idx(model, X_test, y_test,
                                                   seed=None)
    print("Testing:")
    for j in range(5):
      for i, s in enumerate(outputs):
        print(
            f"({s}): Prediction = {predictions[j, i]:.3f}. Truth = {truths[j, i]:.3f}. P - T = {predictions[j, i] - truths[j, i]:.3f}")
        print(f"Inputs: {inputs[j]}")
    with open("./model_predictions.txt", "a", encoding="utf-8") as outfile:
      outfile.write(f"{model_name}\n")
      outfile.write(f"Test MSE = {model_stats[model_name][0]:1.2e}\n")
      outfile.write(f"Test MAE = {model_stats[model_name][1]:1.2e}\n")

  return models, model_stats


def check_random_idx(model, X_test, y_test, seed=42):
  if seed:
    np.random.seed(seed)
  idx = np.random.randint(0, X_test.shape[0])
  predictions = model.predict(X_test[idx:idx+10])
  truths = y_test[idx:idx+10]

  return predictions, truths, X_test[idx:idx+10]


def compile_and_train(config: Config, X_train, y_train, X_val, y_val,
                      model_type="MLP"):
  # Compile a model from 'config', train & validate it (no eval with test)
  # Return the trained model
  input_shape = (config.num_inputs,)
  models = {
      # Models are compiled upon initialisation
      "MLP": mlp(input_shape=input_shape, num_out=config.num_outputs),
      "CNN": cnn_1d(input_shape=(config.num_inputs, 1),
                    num_out=config.num_outputs),
      "EncDec": encoder_decoder(input_shape=input_shape,
                                num_out=config.num_outputs)
  }
  model = models[model_type]
  model_name = f"TF_{model_type}{config.uid}"

  # Train
  # model_stats = {}
  print(f"Training \'{model_name}\'")
  model.summary()
  # TensorFlow training
  checkpoint_path = f"./models/{model_name}.ckpt"
  checkpoint_cb = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                  save_weights_only=True,
                                                  verbose=1)
  history = model.fit(X_train, y_train,
                      batch_size=1024,
                      epochs=500,
                      # epochs=5,
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

  return model, history, model_name


def setup_and_train(config: Config, test_files=None, abundance_mappings=None):
  # TODO:
  # Refactor to have a 'compile_and_train' function that takes the config and
  # model type, compiles & fits model, then returns
  # This might fix the memory issues
  models = prepare_models(config.num_inputs, config.num_outputs, uid=config.uid)
  print("Set up models")
  (X_train, y_train), (X_val, y_val), (X_test, y_test) =\
      prepare_dataset(use_logn=config.use_logn,
                      input_species=config.input_keys,
                      output_species=config.output_keys,
                      test_files=test_files,
                      abundance_mappings=abundance_mappings
                      )
  print("Set up dataset")
  models, model_stats = train_models(models, X_train, y_train, X_val, y_val,
                                     X_test, y_test, config.output_keys)
  print_stats(model_stats)

  return models, model_stats


def prepare_combined_dataset(train_file: str, test_file: str, config: Config):
  # Set up combined dataset
  df_train = vaex.open(train_file)
  df_test = vaex.open(test_file)

  X_keys = [*[f"A_{s}" for s in config.input_keys], "density", "temperature"]
  y_keys = [f"{s}_EQ" for s in config.output_keys]

  df_train, df_test = clean_dfs(df_train, df_test)

  print(list(df_train.columns))
  print("Cleaned DFs")
  check_dfs(df_train, df_test)

  data_train = df_train[X_keys]
  label_train = df_train[y_keys]
  X_test = df_test[X_keys]
  y_test = df_test[y_keys]
  print("Input columns:")
  print(list(data_train.columns))
  print("Output columns:")
  print(list(label_train.columns))

  # Apply log scaling
  data_train["density"] = np.log10(data_train["density"])
  X_test["density"] = np.log10(X_test["density"])
  for k in y_keys:
    label_train[k] = np.log10(label_train[k])
    y_test[k] = np.log10(y_test[k])
  print("Applied log scaling")

  X_trainval = data_train.values
  y_trainval = label_train.values
  X_test = X_test.values
  y_test = y_test.values
  X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval,
                                                    test_size=0.25, random_state=42)
  print("Split data")
  print("Train", X_train.shape, y_train.shape)
  print("Val", X_val.shape, y_val.shape)
  print("Test", X_test.shape, y_test.shape)
  for i in range(X_train.shape[1]):
    print(X_keys[i], minmax(X_train[:, i]))
  scaler = MinMaxScaler(feature_range=(0, 1))
  scaler.fit(X_train)
  X_train = scaler.transform(X_train)
  X_val = scaler.transform(X_val)
  X_test = scaler.transform(X_test)
  # print("Any nans?")
  # for k in y_keys:
  #   print(k, np.any(np.isnan(y_train[k].values)))

  return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def setup_train_combined(train_file: str, test_file: str, config: Config):
  # Load a combined parquet DF for training and another for testing, setup
  # data and train relevant networks
  (X_train, y_train), (X_val, y_val), (X_test, y_test) =\
      prepare_combined_dataset(train_file, test_file, config)
  print("Loaded datasets")
  model_types = [
      # "MLP",
      "CNN",
      # "EncDec"
  ]
  model_stats = {}
  for model_type in model_types:
    # Compile and train model
    model, history, model_name = compile_and_train(config, X_train, y_train, X_val, y_val,
                                                   model_type=model_type)
    # Evaluate on test set
    model_stats[model_name] = model.evaluate(X_test, y_test, verbose=2)
    print(f"Testing {model_name}")
    print_stats(model_stats)
    predictions, truths, inputs = check_random_idx(model, X_test, y_test,
                                                   seed=None)
    for j in range(5):
      for i, s in enumerate(config.output_keys):
        print(
            f"({s}): Prediction = {predictions[j, i]:.3f}. Truth = {truths[j, i]:.3f}. P - T = {predictions[j, i] - truths[j, i]:.3f}")
        print(f"Inputs: {inputs[j]}")
    with open("./model_predictions.txt", "a", encoding="utf-8") as outfile:
      outfile.write(f"{model_name}\n")
      outfile.write(f"Test MSE = {model_stats[model_name][0]:1.2e}\n")
      outfile.write(f"Test MAE = {model_stats[model_name][1]:1.2e}\n")


def main():
  chem1_keys = ["H", "C", "O", "M", "H2", "CH", "OH", "CO"]
  chem1_atomic_keys = ["H", "C", "O", "M"]
  model_ids = [
      "mm00",
      "mm20",
      "mm30",
      "mm30c20n20o20",
      "mm30c20n20o04",
  ]
  uid_suffixes_3d = [f"_{model_id}_3d" for model_id in model_ids]
  test_files_3d = [f"../res/df/d3t63g40mm{id_}chem1_074.parquet" for id_ in
                   [
                       "00",
                       "20",
                       "30",
                       "30c20n20o20",
                       "30c20n20o04"
                   ]]
  abundance_mappings = [
      mm00_abundances,
      mm20a04_abundances,
      mm30a04_abundances,
      mm30a04c20n20o20_abundances,
      mm30a04c20n20o04_abundances
  ]
  configs_3d_chem1 = [Config(6, 8, input_species=chem1_atomic_keys,
                             output_species=chem1_keys, use_logn=False,
                             uid_suffix=uid_suffix) for uid_suffix in uid_suffixes_3d]
  configs = [
      *configs_3d_chem1
      # # All log(n) -> CO
      # Config(10, 1, input_species=chem1_keys,
      #        output_species=["CO"], use_logn=True,
      #        uid_suffix=uid_suffix),
      # # All log(n) -> all log(n)
      # Config(10, 8, input_species=chem1_keys,
      #        output_species=chem1_keys, use_logn=True,
      #        uid_suffix=uid_suffix),
      # Config(6, 1, input_species=chem1_atomic_keys,
      #        output_species=["CO"], use_logn=True,
      #        uid_suffix=uid_suffix),
      # Config(6, 8, input_species=chem1_atomic_keys,
      #        output_species=chem1_keys, use_logn=True,
      #        uid_suffix=uid_suffix),
  ]

  # for (config, test_file, abundance_mapping) in zip(configs, test_files_3d, abundance_mappings):
  #   print(f"Running config {config.uid}")
  #   models, model_stats = setup_and_train(
  #       config, test_files=[test_file], abundance_mappings=[abundance_mapping])

  # Run combined dataset
  res_dir = "../res/df/"
  combined_config_dwarf_cemp = Config(6, 8, input_species=chem1_atomic_keys,
                           output_species=chem1_keys, use_logn=True,
                           uid_suffix="_combined_dwarf_cemp_3d")
  combined_config_rgb = Config(6, 8, input_species=chem1_atomic_keys,
                           output_species=chem1_keys, use_logn=True,
                           uid_suffix="_combined_rgb_3d")
  combined_config_dwarf_rgb = Config(6, 8, input_species=chem1_atomic_keys,
                           output_species=chem1_keys, use_logn=True,
                           uid_suffix="_combined_dwarf_cemp_rgb_3d")
  combined_config_dwarf_rgb_abu = Config(6, 8, input_species=chem1_atomic_keys,
                           output_species=chem1_keys, use_logn=False,
                           uid_suffix="_combined_dwarf_cemp_rgb_3d")
  # setup_train_combined(f"{res_dir}/combined_dwarf_cemp_074.parquet",
  #                      f"{res_dir}/combined_dwarf_cemp_116.parquet",
  #                      config=combined_config_dwarf_cemp)
  # setup_train_combined(f"{res_dir}/combined_rgb_074.parquet",
  #                      f"{res_dir}/combined_rgb_116.parquet",
  #                      config=combined_config_rgb)
  # setup_train_combined(f"{res_dir}/combined_dwarf_cemp_rgb_074.parquet",
  #                      f"{res_dir}/combined_dwarf_cemp_rgb_116.parquet",
  #                      config=combined_config_dwarf_rgb)
  setup_train_combined(f"{res_dir}/combined_dwarf_cemp_rgb_074.parquet",
                       f"{res_dir}/combined_dwarf_cemp_rgb_116.parquet",
                       config=combined_config_dwarf_rgb_abu)


if __name__ == "__main__":
  main()

"""
TODO:
  - use combined set 74 for training and combined set 116 for testing, DFs are
    already init'd
    - can write a few new functions to do this
"""
