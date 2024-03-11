# Generate figures to compare predictions with GT for:
# 1. random xz slice
# 2. random xy slice
# 3. entire test data for each species
import numpy as np
import tensorflow as tf
import os
from typing import List
import re

from nn_models import mlp, cnn_1d, encoder_decoder
    
SEED = 420
np.random.seed(SEED)
tf.random.set_seed(SEED)


def model_creator(model_type: str, num_in: int, num_out: int):
    model_types = {
        "TF_MLP": lambda n_i, n_o: mlp(input_shape=(n_i,), num_out=n_o),
        "TF_CNN": lambda n_i, n_o: cnn_1d(input_shape=(n_i, 1), num_out=n_o),
        "TF_EncDec": lambda n_i, n_o: encoder_decoder(input_shape=(n_i,), num_out=n_o)
    }
    return model_types[model_type](num_in, num_out)


def get_checkpoints(data_dir: str, dataset: str) -> List[str]:
    checkpoints = [f for f in os.listdir(data_dir) if f.endswith(".ckpt.index") and dataset in f]
    checkpoints = [c.replace(".ckpt.index", "") for c in checkpoints]

    return checkpoints

def parse_ckpt_str(ckpt_str: str):
    # Parse checkpoint name to get input shape and output shape
    pattern = re.compile(r'([A-Za-z]+(?:_[A-Za-z]+)?)_([0-9]+)-([0-9]+)_([A-Za-z]+)')

    # Example usage
    text = "TF_MLP6-8_abu_combined_rgb_3d"
    match = pattern.match(text)
    print(match)

    if match:
        result = list(match.groups())
        print(result)
        return result

    # match = re.match(r'([A-Z_]+)_([A-Za-z0-9]+)([0-9]+)-([0-9]+)_([a-z_]+)_', ckpt_str)
    # print(ckpt_str)
    
    # if match:
    #     model_type = match.group(1)
    #     num_in = int(match.group(2))
    #     num_out = int(match.group(3))
    #     abu_id = match.group(5)
    #     use_logn = abu_id == "logn"
        
    #     return [model_type, num_in, num_out, use_logn]
    # else:
    #     return None


def main():
    data_dir = "./models"
    dataset = "combined_rgb"
    test_snap_num = "116"
    checkpoints = get_checkpoints(data_dir, dataset)
    for ckpt_str in checkpoints:
        # Load model
        model_type, num_in, num_out, use_logn = parse_ckpt_str(ckpt_str)
        model = model_creator(model_type, num_in, num_out)
        model.load_weights(f"{data_dir}/{ckpt_str}")
        model.summary()
        # Load test data, use either logn or abu

if __name__ == "__main__":
    main()