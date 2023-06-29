# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 12:38:41 2023

@author: sasa5
"""
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import time
from datasets import Dataset
from transformers import DataCollatorWithPadding
import evaluate
from transformers import  TrainingArguments, Trainer
from tqdm import tqdm
import torch

def batch_predict(model, data, batch_size=32):
    model.eval()  # put model in evaluation mode
    batched_data = []
    
    # Generate batches
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        batched_data.append(batch)
    
    predictions = []

    start_time = time.time()  # Record start time
    for batch in tqdm(batched_data, desc="Predicting"):  # Add progress bar
        encoded_data = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(encoded_data['input_ids'], attention_mask=encoded_data['attention_mask'])
            pred_classes = torch.argmax(outputs.logits, dim=1)
            predictions.extend(pred_classes.cpu().numpy())  # Move prediction to CPU and convert to numpy array

    end_time = time.time()  # Record end time
    print(f"Prediction completed in {end_time - start_time} seconds.")  # Print elapsed time

    return predictions

def preprocess_function(examples):
    return tokenizer(examples["text"], max_length=maxlen,truncation=True, padding=True)
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

ROOT_DIR = ""
RES_DIR = os.path.join(ROOT_DIR, "resources")
MOD_DIR = os.path.join(ROOT_DIR, "ml_models")
TRAIN_DIR = os.path.join(ROOT_DIR, "train_sets")
REP_DIR = os.path.join(ROOT_DIR, "reports", "Transformer")
maxlen = 200

# Record the start time
start_time = time.time()
i = 6   #This is iteration, becasue time needed to fit model 
        #it's not place in loop

polarity = "POS" # same reason for polarity
BUFFER_SIZE = 1000
BATCH_SIZE = 128

# File name
name = f"LM{polarity}{i}.csv"

# Read the data from the CSV file
X = pd.read_csv(os.path.join(TRAIN_DIR, f"X_train_{name}"))["Sysnet"]
y = pd.read_csv(os.path.join(TRAIN_DIR, f"y_train_{name}"))[polarity]

X_test = pd.read_csv(os.path.join(TRAIN_DIR, f"X_test_{name}"))["Sysnet"]
y_test = pd.read_csv(os.path.join(TRAIN_DIR, f"y_test_{name}"))[polarity]

X.rename("text", inplace=True)
y.rename("labels", inplace=True)

# Split dataset into training and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, 
                                                  stratify=y, random_state=42)
id2label = {0: "NON-POSITIVE", 1: "POSITIVE"}
label2id = {"NON-POSITIVE": 0, "POSITIVE": 1}

# #just first time to load base model
model = "classla/bcms-bertic"
# model = f"Tanor/BERTic_{i}_{polarity}"

tokenizer = AutoTokenizer.from_pretrained(model)
model_POS = AutoModelForSequenceClassification.from_pretrained(
    model, num_labels=2,  id2label=id2label, 
    label2id=label2id, )
#JUst first time to upload base model for fitting 
model = f"Tanor/BERTic_{i}_{polarity}"
# model_POS.push_to_hub(model)
# tokenizer.push_to_hub(model)
dataset_val = Dataset.from_pandas(pd.concat([X_val, y_val], axis=1))
dataset_train = Dataset.from_pandas(pd.concat([X_train, y_train], axis=1))

tokenised_val=dataset_val.map(preprocess_function)
tokenised_train =dataset_train.map(preprocess_function)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
accuracy = evaluate.load("accuracy")
training_args = TrainingArguments(
    output_dir=os.path.join("C:",model),
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=True,
)

trainer = Trainer(
    model=model_POS,
    args=training_args,
    train_dataset=tokenised_train,
    eval_dataset=tokenised_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()


predicted_classes = batch_predict(model_POS, list(X_test), batch_size=32)
y_test_np = y_test.values
confusion_mat = confusion_matrix(y_test_np, predicted_classes)

print(confusion_mat)
classification_rep = classification_report(y_test_np, predicted_classes)

print(classification_rep)
