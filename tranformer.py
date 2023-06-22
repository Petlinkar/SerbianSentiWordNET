# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 12:27:05 2023

@author: sasa5
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras import layers, Sequential
from tensorflow import keras

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

# Create directory if not exists
if not os.path.exists(REP_DIR):
    os.makedirs(REP_DIR)

# Define constants
i = 0
polarity = "POS"
BUFFER_SIZE = 1000
BATCH_SIZE = 128
VOCAB_SIZE = 25000
maxlen = 200  # Only consider the first 200 words

# File name
name = f"LM{polarity}{i}.csv"

# Read the data from the CSV file
X = pd.read_csv(os.path.join(TRAIN_DIR, f"X_train_{name}"))["Sysnet"]
y = pd.read_csv(os.path.join(TRAIN_DIR, f"y_train_{name}"))[polarity]

X_test = pd.read_csv(os.path.join(TRAIN_DIR, f"X_test_{name}"))["Sysnet"]
y_test = pd.read_csv(os.path.join(TRAIN_DIR, f"y_test_{name}"))[polarity]

# Split dataset into training and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, stratify=y)

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
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
validation_dataset = validation_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Define encoder for text vectorization
encoder = tf.keras.layers.TextVectorization(standardize=None,
    max_tokens=VOCAB_SIZE, output_mode="int")

# Train encoder on the text data
encoder.adapt(train_dataset.map(lambda text, label: text))

# Define hyperparameters
embed_dim = 64  # Embedding size for each token
num_heads = 4  # Number of attention heads
ff_dim = 64  # Hidden layer size in feed forward network inside transformer

# Define the model architecture using Keras Functional API
inputs = layers.Input(shape=(1,), dtype=tf.string)  # Input layer for raw string data
embedding_layer = TokenAndPositionEmbedding(maxlen, VOCAB_SIZE, embed_dim)  # Token and position embedding layer
x = encoder(inputs)  # Text encoding layer
x = embedding_layer(x)  # Apply embedding to the encoded input

transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)  # Transformer block layer
x = transformer_block(x)  # Apply the transformer block to the embeddings

x = layers.GlobalAveragePooling1D()(x)  # Apply global average pooling
x = layers.Dropout(0.1)(x)  # Apply dropout for regularization
x = layers.Dense(20, activation="relu")(x)  # Dense layer with ReLU activation

# Uncomment below if additional dropout layer is desired
# x = layers.Dropout(0.1)(x)

outputs = layers.Dense(1, activation='sigmoid')(x)  # Output layer with sigmoid activation

model_POS = keras.Model(inputs=inputs, outputs=outputs)  # Define the model

model_POS.compile(
    optimizer="adam", loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), metrics=["binary_accuracy"]
)
history = model_POS.fit(train_dataset, epochs=5,
                    validation_data=validation_dataset,
                    validation_steps=5)
y_pred = model_POS.predict(X_test)


# Assuming y_pred is continuous value from 0 to 1, we need to convert it to binary.
# Usually, we use a threshold of 0.5 for this conversion
y_pred_binary = np.where(y_pred > 0.5, 1, 0)

# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_binary)
print('Confusion Matrix:\n', conf_matrix)

# Calculate the classification report
class_report = classification_report(y_test, y_pred_binary)
print('Classification Report:\n', class_report)
with open(REP_DIR + "report_" + name + ".txt", "w") as f:
    f.write(str(conf_matrix))
    f.write("\n\n")
    f.write(class_report)


# Assuming you have a list X that contains your test inputs, and that your model was trained on
# You would convert the predictions to a binary output similar to y_pred_binary
predicted_y = y_pred_binary
X_test_str = [seq.decode('utf-8') for seq in X_test.numpy()]
# Create a data frame
table = pd.DataFrame({"X": X_test_str, "Predicted": predicted_y.flatten(), "Real": y_test.numpy().flatten()})

# Create a table of misclassified X values.
misclassified_X = table[table["Predicted"] != table["Real"]]

# Save the table to a file.
misclassified_X.to_csv(REP_DIR + "table_" + name, index=False)


polarity = "NEG"
# File name
name = f"LM{polarity}{i}.csv"

# Read the data from the CSV file
X = pd.read_csv(os.path.join(TRAIN_DIR, f"X_train_{name}"))["Sysnet"]
y = pd.read_csv(os.path.join(TRAIN_DIR, f"y_train_{name}"))[polarity]

X_test = pd.read_csv(os.path.join(TRAIN_DIR, f"X_test_{name}"))["Sysnet"]
y_test = pd.read_csv(os.path.join(TRAIN_DIR, f"y_test_{name}"))[polarity]

# Split dataset into training and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, stratify=y)

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
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
validation_dataset = validation_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Define encoder for text vectorization
encoder = tf.keras.layers.TextVectorization(standardize=None,
    max_tokens=VOCAB_SIZE, output_mode="int")

# Train encoder on the text data
encoder.adapt(train_dataset.map(lambda text, label: text))

# Define hyperparameters
embed_dim = 64  # Embedding size for each token
num_heads = 4  # Number of attention heads
ff_dim = 64  # Hidden layer size in feed forward network inside transformer

# Define the model architecture using Keras Functional API
inputs = layers.Input(shape=(1,), dtype=tf.string)  # Input layer for raw string data
embedding_layer = TokenAndPositionEmbedding(maxlen, VOCAB_SIZE, embed_dim)  # Token and position embedding layer
x = encoder(inputs)  # Text encoding layer
x = embedding_layer(x)  # Apply embedding to the encoded input

transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)  # Transformer block layer
x = transformer_block(x)  # Apply the transformer block to the embeddings

x = layers.GlobalAveragePooling1D()(x)  # Apply global average pooling
x = layers.Dropout(0.1)(x)  # Apply dropout for regularization
x = layers.Dense(20, activation="relu")(x)  # Dense layer with ReLU activation

# Uncomment below if additional dropout layer is desired
# x = layers.Dropout(0.1)(x)

outputs = layers.Dense(1, activation='sigmoid')(x)  # Output layer with sigmoid activation

model_NEG = keras.Model(inputs=inputs, outputs=outputs)  # Define the model

model_NEG.compile(
    optimizer="adam", loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), metrics=["binary_accuracy"]
)
history = model_NEG.fit(train_dataset, epochs=5,
                    validation_data=validation_dataset,
                    validation_steps=5)
y_pred = model_NEG.predict(X_test)


# Assuming y_pred is continuous value from 0 to 1, we need to convert it to binary.
# Usually, we use a threshold of 0.5 for this conversion
y_pred_binary = np.where(y_pred > 0.5, 1, 0)

# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_binary)
print('Confusion Matrix:\n', conf_matrix)

# Calculate the classification report
class_report = classification_report(y_test, y_pred_binary)
print('Classification Report:\n', class_report)
with open(REP_DIR + "report_" + name + ".txt", "w") as f:
    f.write(str(conf_matrix))
    f.write("\n\n")
    f.write(class_report)


# Assuming you have a list X that contains your test inputs, and that your model was trained on
# You would convert the predictions to a binary output similar to y_pred_binary
predicted_y = y_pred_binary
X_test_str = [seq.decode('utf-8') for seq in X_test.numpy()]
# Create a data frame
table = pd.DataFrame({"X": X_test_str, "Predicted": predicted_y.flatten(), "Real": y_test.numpy().flatten()})

# Create a table of misclassified X values.
misclassified_X = table[table["Predicted"] != table["Real"]]

# Save the table to a file.
misclassified_X.to_csv(REP_DIR + "table_" + name, index=False)