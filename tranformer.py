# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 12:27:05 2023

@author: Petalinkar Sasa

This script performs sentiment analysis using a transformer model.
It loads and preprocesses the data, trains the model, evaluates its performance,
and saves a report with the classification metrics and the misclassified examples.

The script is designed to be used with a specific data format and directory structure,
and it may need to be adapted for other use cases.
"""


import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

def polarity_correction(pos, neg):
    """
    This function corrects the polarity of the output of a two-output model 
    that outputs a positive sentiment score and a negative sentiment score. 

    Args:
        pos (tensor): Tensor with positive sentiment scores.
        neg (tensor): Tensor with negative sentiment scores.

    Returns:
        tuple: A tuple with tensors of corrected positive and negative sentiment scores.
    """
    one = tf.convert_to_tensor(1.0)  # Tensor of 1.0 for calculations

    # Subtract negative sentiment score from 1 and multiply it with the positive sentiment score
    ret_pos = pos * (one - neg)

    # Subtract positive sentiment score from 1 and multiply it with the negative sentiment score
    ret_neg = neg * (one - pos)

    return ret_pos, ret_neg

class TransformerBlock(layers.Layer):
    """
    This class defines the transformer block which is the main building block
    of a transformer network. It includes two main parts:
    1. Multi-head self-attention mechanism,
    2. Position-wise fully connected feed-forward network.
    """
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        """
        Initializes the TransformerBlock with the parameters to be used.

        Args:
            embed_dim (int): Dimension of the embedding.
            num_heads (int): Number of attention heads.
            ff_dim (int): Hidden layer size in feed forward network inside transformer.
            rate (float, optional): Dropout rate. Defaults to 0.1.
        """
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)  # Multi-head attention layer
        self.ffn = Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]  # Feed-forward network
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)  # Layer normalization after multi-head attention
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)  # Layer normalization after feed-forward network
        self.dropout1 = layers.Dropout(rate)  # Dropout for regularization after multi-head attention
        self.dropout2 = layers.Dropout(rate)  # Dropout for regularization after feed-forward network

    def call(self, inputs, training):
        """
        Method for the forward pass in the transformer block.

        Args:
            inputs (tensor): Input tensor.
            training (bool): Whether the model is in training mode.

        Returns:
            tensor: Output tensor after passing through the transformer block.
        """
        attn_output = self.att(inputs, inputs)  # Compute self-attention
        attn_output = self.dropout1(attn_output, training=training)  # Apply dropout
        out1 = self.layernorm1(inputs + attn_output)  # Add & normalize
        ffn_output = self.ffn(out1)  # Compute feed-forward network
        ffn_output = self.dropout2(ffn_output, training=training)  # Apply dropout
        return self.layernorm2(out1 + ffn_output)  # Add & normalize
class TokenAndPositionEmbedding(layers.Layer):
    """
    This class implements the token and positional embedding layer, which
    combines the embeddings of the tokens and their corresponding positions
    to provide the Transformer model with information about the order of the
    tokens in the input sequence.
    """
    def __init__(self, maxlen, vocab_size, embed_dim):
        """
        Initializes the TokenAndPositionEmbedding with the parameters to be used.

        Args:
            maxlen (int): The maximum possible length for the input sequences.
            vocab_size (int): The size of the vocabulary in the text data.
            embed_dim (int): The dimension of the embedding vectors.
        """
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)  # Token embedding layer
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)  # Positional embedding layer

    def call(self, x):
        """
        Method for the forward pass in the token and positional embedding layer.

        Args:
            x (tensor): Input tensor with tokenized words.

        Returns:
            tensor: Output tensor after adding token and positional embeddings.
        """
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)  # Generate position sequences
        positions = self.pos_emb(positions)  # Compute positional embeddings
        x = self.token_emb(x)  # Compute token embeddings
        return x + positions  # Add token embeddings with positional embeddings

# Define directories
ROOT_DIR = ""
RES_DIR = os.path.join(ROOT_DIR, "resources")
MOD_DIR = os.path.join(ROOT_DIR, "ml_models")
TRAIN_DIR = os.path.join(ROOT_DIR, "train_sets")
REP_DIR = os.path.join(ROOT_DIR, "reports", "Transformer")

# Define constants
BUFFER_SIZE = 1000
BATCH_SIZE = 128
VOCAB_SIZE = 25000
MAX_LEN = 300  # Only consider the first 300 words
EMBED_DIM = 64  # Embedding size for each token
NUM_HEADS = 4  # Number of attention heads
FF_DIM = 64  # Hidden layer size in feed forward network inside transformer
N_TRANSFORMER_LAYERS = 1  # Number of transformer layers
DATASET_ITERATIONS = [0, 2, 4, 6]  # Dataset iterations to process

# Create directory if not exists
if not os.path.exists(REP_DIR):
    os.makedirs(REP_DIR)

# Set a seed for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
def load_and_preprocess_data(polarity, iteration):
    """
    Load and preprocess the data for a given polarity and iteration.

    Args:
        polarity (str): The polarity of the sentiment ("POS" or "NEG").
        iteration (int): The iteration number.

    Returns:
        tuple: A tuple with the training and validation datasets.
    """
    # File name
    name = f"LM{polarity}{iteration}.csv"

    # Read the data from the CSV file
    X = pd.read_csv(os.path.join(TRAIN_DIR, f"X_train_{name}"))["Sysnet"]
    y = pd.read_csv(os.path.join(TRAIN_DIR, f"y_train_{name}"))[polarity]

    X_test = pd.read_csv(os.path.join(TRAIN_DIR, f"X_test_{name}"))["Sysnet"]
    y_test = pd.read_csv(os.path.join(TRAIN_DIR, f"y_test_{name}"))[polarity]

    # Split dataset into training and validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, stratify=y, random_state=SEED)

    # Convert pandas dataframes to tensorflow tensors
    X_train = tf.convert_to_tensor(X_train, name="Definicija")
    y_train = tf.convert_to_tensor(y_train, name="Sentiment")

    X_val = tf.convert_to_tensor(X_val, name="Definicija")
    y_val = tf.convert_to_tensor(y_val, name="Sentiment")

    X_test = tf.convert_to_tensor(X_test, name="Definicija")
    y_test = tf.convert_to_tensor(y_test, name="Sentiment")

    # Create tf.data datasets for training and validation
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    validation_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))

    # Shuffle, batch, and prefetch data for performance
    train_dataset = train_dataset.shuffle(BUFFER_SIZE, seed=SEED).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_dataset, validation_dataset, X_test, y_test

def create_model(n_transformer_layers=1):
    """
    Create and compile the transformer model.

    Args:
        n_transformer_layers (int, optional): The number of transformer layers. Default is 1.

    Returns:
        keras.Model: The compiled model.
    """
    # Define the model architecture using Keras Functional API
    inputs = layers.Input(shape=(1,), dtype=tf.string)  # Input layer for raw string data
    embedding_layer = TokenAndPositionEmbedding(MAX_LEN, VOCAB_SIZE, EMBED_DIM)  # Token and position embedding layer
    x = encoder(inputs)  # Text encoding layer
    x = embedding_layer(x)  # Apply embedding to the encoded input

    for _ in range(n_transformer_layers):  # Add n_transformer_layers transformer blocks
        transformer_block = TransformerBlock(EMBED_DIM, NUM_HEADS, FF_DIM)  # Transformer block layer
        x = transformer_block(x)  # Apply the transformer block to the embeddings

    x = layers.GlobalAveragePooling1D()(x)  # Apply global average pooling
    x = layers.Dropout(0.1, seed=SEED)(x)  # Apply dropout for regularization
    x = layers.Dense(20, activation="relu")(x)  # Dense layer with ReLU activation

    outputs = layers.Dense(1, activation='sigmoid')(x)  # Output layer with sigmoid activation

    model = keras.Model(inputs=inputs, outputs=outputs)  # Define the model

    model.compile(optimizer="adam", 
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), 
                  metrics=["binary_accuracy"])

    return model
def train_model(model, train_dataset, validation_dataset):
    """
    Train the model on the given datasets.

    Args:
        model (keras.Model): The compiled model.
        train_dataset (tf.data.Dataset): The training dataset.
        validation_dataset (tf.data.Dataset): The validation dataset.

    Returns:
        History: The history object returned by model.fit().
    """
    history = model.fit(train_dataset, 
                        epochs=5, 
                        validation_data=validation_dataset, 
                        validation_steps=5)

    return history

def evaluate_model(model, X_test):
    """
    Evaluate the model on the test data.

    Args:
        model (keras.Model): The trained model.
        X_test (Tensor): The test input data.

    Returns:
        np.array: The predicted outputs.
    """
    y_pred = model.predict(X_test)

    # Assuming y_pred is a continuous value from 0 to 1, we need to convert it to binary.
    # Usually, we use a threshold of 0.5 for this conversion
    y_pred_binary = np.where(y_pred > 0.5, 1, 0)

    return y_pred_binary

def write_report(y_test, y_pred_binary, polarity, iteration):
    """
    Write the classification report and confusion matrix to a text file.

    Args:
        y_test (Tensor): The true labels.
        y_pred_binary (np.array): The predicted labels.
        polarity (str): The polarity of the sentiment ("POS" or "NEG").
        iteration (int): The iteration number.
    """
    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred_binary)
    print('Confusion Matrix:\n', conf_matrix)

    # Calculate the classification report
    class_report = classification_report(y_test, y_pred_binary)
    print('Classification Report:\n', class_report)

    # File name
    name = f"LM{polarity}{iteration}.csv"

    # Writing report
    with open(os.path.join(REP_DIR, f"report_{name}.txt"), "w") as f:
        f.write(str(conf_matrix))
        f.write("\n\n")
        f.write(class_report)
def save_misclassified(X_test, y_pred_binary, y_test, polarity, iteration):
    """
    Save the misclassified examples to a CSV file.

    Args:
        X_test (Tensor): The test input data.
        y_pred_binary (np.array): The predicted labels.
        y_test (Tensor): The true labels.
        polarity (str): The polarity of the sentiment ("POS" or "NEG").
        iteration (int): The iteration number.
    """
    # Convert tensors to pandas DataFrames for easier manipulation
    X_test_df = pd.DataFrame(X_test.numpy(), columns=["Sysnet"])
    y_test_df = pd.DataFrame(y_test.numpy(), columns=[polarity])

    # Concatenate the two DataFrames along the columns
    test_data = pd.concat([X_test_df, y_test_df], axis=1)

    # Add the predicted labels to the DataFrame
    test_data["Predicted"] = y_pred_binary

    # Select the misclassified examples
    misclassified = test_data[test_data[polarity] != test_data["Predicted"]]

    # File name
    name = f"LM{polarity}{iteration}.csv"

    # Save the misclassified examples to a CSV file
    misclassified.to_csv(os.path.join(REP_DIR, f"misclassified_{name}"), index=False)

def main():
    """
    The main function to execute the script.
    """
    # Record the start time for the total execution
    total_start_time = time.time()

    # Repeat the process for each dataset iteration and polarity
    for i in DATASET_ITERATIONS:
        # Record the start time for this iteration
        iter_start_time = time.time()

        for polarity in ["POS", "NEG"]:
            train_dataset, validation_dataset, X_test, y_test = load_and_preprocess_data(polarity, i)

            # Create and train the model
            model = create_model(N_TRANSFORMER_LAYERS)
            train_model(model, train_dataset, validation_dataset)

            # Evaluate the model
            y_pred_binary = evaluate_model(model, X_test)

            # Write the report and save the misclassified examples
            write_report(y_test, y_pred_binary, polarity, i)
            save_misclassified(X_test, y_pred_binary, y_test, polarity, i)

        # Record the end time for this iteration
        iter_end_time = time.time()

        # Calculate the execution time for this iteration
        iter_total_time = iter_end_time - iter_start_time

        # Convert the iteration time to hours, minutes, and seconds
        hours, rem = divmod(iter_total_time, 3600)
        minutes, seconds = divmod(rem, 60)

        # Save the timing result for this iteration in a text file
        with open(os.path.join(REP_DIR, f'execution_time_{i}.txt'), 'w') as f:
            f.write("Execution Time for Iteration {}: {:0>2}:{:0>2}:{:05.2f}".format(i, int(hours), int(minutes), seconds))

    # Record the end time for the total execution
    total_end_time = time.time()

    # Calculate the total execution time
    total_total_time = total_end_time - total_start_time

    # Convert the total time to hours, minutes, and seconds
    hours, rem = divmod(total_total_time, 3600)
    minutes, seconds = divmod(rem, 60)

    # Save the total timing result in a text file
    with open(os.path.join(REP_DIR, 'execution_time_total.txt'), 'w') as f:
        f.write("Total Execution Time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

if __name__ == "__main__":
    main()
