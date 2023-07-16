import torch
from transformers import pipeline
import pandas as pd
import os
ROOT_DIR = ""
RES_DIR = os.path.join(ROOT_DIR, "resources")
MOD_DIR = os.path.join(ROOT_DIR, "ml_models")
TRAIN_DIR = os.path.join(ROOT_DIR, "train_sets")
REP_DIR = os.path.join(ROOT_DIR, "reports", "CompareLLM")

def print_correctly_classified_instances(i, polarity, model ="BERTic"):
    # Empty GPU cache before testing model
    torch.cuda.empty_cache()

    # Construct file name
    name = f"UP{polarity}{i}.csv"
    
    # Load the test data
    X_test = pd.read_csv(os.path.join(TRAIN_DIR, f"X_test_{name}"))["Sysnet"]
    y_test = pd.read_csv(os.path.join(TRAIN_DIR, f"y_test_{name}"))[polarity]
    X_test = X_test.fillna("")        
    # Construct model name
    if (model=="BERTic"):
        model_name = f"Tanor/BERTicSENT{polarity}{i}"
    if (model=="BERTicovo"):
        model_name = f"Tanor/BERTicovoSENT{polarity}{i}"
    if (model=="SRBGPT"):
        model_name = f"Tanor/SRGPTSENT{polarity}{i}"
    
    # Load model using pipeline
    pipe = pipeline("text-classification", model=model_name)
    
    # Define label to id mapping
    label2id = {"NON-POSITIVE": 0, "POSITIVE": 1}
    if (polarity =="NEG"):
        label2id = {"NON-NEGATIVE": 0, "NEGATIVE": 1}
    
    # Process test data through pipeline
    data = pipe(X_test.to_list())
    
    # Convert the list of dictionaries into a pandas DataFrame
    df = pd.DataFrame(data)
    
    # Convert the 'label' column into a series where 'NON-POSITIVE' is 0 and 'POSITIVE' is 1
    df['label'] = df['label'].map(label2id)
    
    # Convert 'label' column into a series
    series = df['label']
    predicted_classes = series.values
    
    # Create a DataFrame with test data, predicted classes and real classes
    table = pd.DataFrame({"X": X_test, "Predicted": predicted_classes, 
                          "Real": y_test})
    
    # Create a table of instances where the predicted class is 1 and the real class is also 1
    correct_class_1 = table[(table["Predicted"] == 1) & (table["Real"] == 1)]
    
    # Print the instances where both the predicted class and real class are 1
    print(correct_class_1["X"])
    
    # Delete the pipeline to free up memory
    del pipe
    torch.cuda.empty_cache()
    
def compare_models(i, polarity, model1, model2):
    # Construct file name
    name = f"UP{polarity}{i}.csv"
    
    # Load the test data
    X_test = pd.read_csv(os.path.join(TRAIN_DIR, f"X_test_{name}"))["Sysnet"].fillna("")
    y_test = pd.read_csv(os.path.join(TRAIN_DIR, f"y_test_{name}"))[polarity]
        
    # Define label to id mapping
    label2id = {"NON-POSITIVE": 0, "POSITIVE": 1} if polarity != "NEG" else {"NON-NEGATIVE": 0, "NEGATIVE": 1}

    # Construct model names
    model_name1 = f"Tanor/{model1}SENT{polarity}{i}"
    model_name2 = f"Tanor/{model2}SENT{polarity}{i}"
    
    # Create dataframes for each model to store the results
    correct_class_1_model1 = run_model(X_test, y_test, model_name1, label2id)
    correct_class_1_model2 = run_model(X_test, y_test, model_name2, label2id)
    
    # Find texts correctly classified by both models
    correct_both = pd.merge(correct_class_1_model1, correct_class_1_model2, how='inner', on=['X'])
    # Find texts correctly classified by the first model but not the second
    correct_model1_only = correct_class_1_model1[~correct_class_1_model1.X.isin(correct_class_1_model2.X)]
    # Find texts correctly classified by the second model but not the first
    correct_model2_only = correct_class_1_model2[~correct_class_1_model2.X.isin(correct_class_1_model1.X)]
    
    print("Texts correctly classified by both models:\n", correct_both["X"])
    print("Texts correctly classified by the first model but not the second:\n", correct_model1_only["X"])
    print("Texts correctly classified by the second model but not the first:\n", correct_model2_only["X"])
    
def run_model(X_test, y_test, model_name, label2id):
    # Empty GPU cache before testing model
    torch.cuda.empty_cache()
    
    # Load model using pipeline
    pipe = pipeline("text-classification", model=model_name)
    
    # Process test data through pipeline
    data = pipe(X_test.to_list())
    
    # Convert the list of dictionaries into a pandas DataFrame
    df = pd.DataFrame(data)
    
    # Convert the 'label' column into a series where 'NON-POSITIVE' is 0 and 'POSITIVE' is 1
    df['label'] = df['label'].map(label2id)
    
    # Create a DataFrame with test data, predicted classes, and real classes
    table = pd.DataFrame({"X": X_test, "Predicted": df['label'], 
                          "Real": y_test})
    
    # Create a table of instances where the predicted class is 1 and the real class is also 1
    correct_class_1 = table[(table["Predicted"] == 1) & (table["Real"] == 1)]
    
    # Delete the pipeline to free up memory
    del pipe
    torch.cuda.empty_cache()

    return correct_class_1

