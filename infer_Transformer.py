from srpskiwordnet import SrbWordNetReader
import pandas as pd
import os
import tensorflow as tf
import numpy as np

from train_tranformer import TokenAndPositionEmbedding, TransformerBlock, evaluate_model, load_and_preprocess_data, write_report, save_misclassified

ROOT_DIR = ""
RES_DIR = os.path.join(ROOT_DIR, "resources")
MOD_DIR = os.path.join(ROOT_DIR, "ml_models")
TRAIN_DIR = os.path.join(ROOT_DIR, "train_sets")
REP_DIR = os.path.join(ROOT_DIR, "reports", "Transformer")

# dictionaty with custom layers
custom_objects = {
    "TokenAndPositionEmbedding": TokenAndPositionEmbedding,
    "TransformerBlock": TransformerBlock
}
DATASET_ITERATIONS = [0, 2, 4, 6]  # Dataset iterations to process
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
def load_model(name):
    """
    Function that loads a model from the MOD_DIR directory
    :param name: name of the model to load
    :return: loaded model
    """
    model_path = os.path.join(MOD_DIR, name)
    # Load architecture
    with open(f'{model_path}.json', 'r') as json_file:
        json_config = json_file.read()
    model = tf.keras.models.model_from_json(json_config)

    # Load weights
    model.load_weights(f'{model_path}.ckpt')
    
    return model

def main():
    """
    The main function to execute the script.
    """

    # Repeat the process for each dataset iteration and polarity
    for i in DATASET_ITERATIONS:

        for polarity in ["POS", "NEG"]:
            _, _, X_test, y_test = load_and_preprocess_data(polarity, i)
            model_name = f"transformer_model_{polarity}_{i}.tf"
            model_path = os.path.join(MOD_DIR, model_name)
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            print("Evaluate on test data")
            results = model.evaluate(X_test, y_test, batch_size=128)
            print("test loss, test acc:", results)
            # Evaluate the model
            y_pred_binary = evaluate_model(model, X_test)

            # Write the report and save the misclassified examples
            write_report(y_test, y_pred_binary, polarity, i)
            save_misclassified(X_test, y_pred_binary, y_test, polarity, i)
if __name__ == "__main__":
    main()

