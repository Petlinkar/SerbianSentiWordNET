# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 11:12:39 2023

@author: Korisnik
"""
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
from copy import deepcopy
from sklearn.metrics import classification_report

#Constansts
TRAIN_DIR = ".//train_sets//"
RES_DIR = ".\\resources\\"
REP_DIR = ".\\reports\\"


def load_file_into_list(filename):
    with open(filename, mode="r", encoding="utf-16") as file:
        lines = [line.strip() for line in file]
    return lines
def train_pipe(pipe, name):
  """Trains a pipeline on data from a given polarity set.

  Args:
    pipe: A scikit-learn pipeline.
    name: The name of the polarity set.

  Returns:
    The trained pipeline.
  """

  # Read the data from the CSV file.
  X = pd.read_csv(TRAIN_DIR + "X_train_" + name)["Sysnet"]
  y = pd.read_csv(TRAIN_DIR + "y_train_" + name)["POS"]

  # Clone the pipeline using the deepcopy function.
  ret_pipe = deepcopy(pipe)

  # Fit the pipeline to the data.
  ret_pipe.fit(X, y)

  # Return the trained pipeline.
  return ret_pipe
def test_pipe(pipe, name):
  """Test trained pipeline on data from a given polarity set.
      Saves reports in REP_DIR folder under report_(name).txt
      Report contains gerated clasifcation report
      and all X values that misslasified along with 
      boy prediscted and real y, as table


  Args:
    pipe: A scikit-learn pipeline.
    name: The name of the polarity set.

  Returns:
    None.
  """

  # Read the data from the CSV file.
  X = pd.read_csv(TRAIN_DIR + "X_test_" + name)["Sysnet"]
  y = pd.read_csv(TRAIN_DIR + "y_test_" + name)["POS"]
  # Create a variable for the predicted values.
  y_predicted = pipe.predict(X)
  
  # Create a classification report.
  report = classification_report(y, y_predicted)

  # Save the report to a file.
  with open(REP_DIR + "report_" + name + ".txt", "w") as f:
    f.write(report)
  # Create a table of misclassified X values.
  table = pd.DataFrame({"X": X, "Predicted": pipe.predict(X), "Real": y})
  # Create a table of misclassified X values.
  misclassified_X = table[table["Predicted"] != table["Real"]]

  # Save the table to a file.
  misclassified_X.to_csv(REP_DIR + "table_" + name, index=False)    
  return None

stop_words3 = load_file_into_list(RES_DIR + "stopwordsSRB tf.txt")
stop_words3 = stop_words3 + ['daljem', 'izgledalo', 'izgledu', 'međuvremenu',
                             'sc', 'sl', 'slučaju', 'tekstu', 'vreme']

PAR = {
    "svm__C": 10000000000000.0,
    "svm__cache_size": 1000,
    "svm__class_weight": "balanced",
    "svm__gamma": 1e-08,
    "svm__kernel": "rbf",
    "tfidf__use_idf": True,
    "vect__ngram_range": (1, 1),
    "vect__stop_words": stop_words3,
}

pipe = Pipeline(
    [
        ("vect", HashingVectorizer()),
        ("tfidf", TfidfTransformer()),

        ("svm", SVC()),

        
    ],
    memory="F:\\Temp\\ML_Cache"
)
pipe.set_params(**PAR)
pipes = {}
for i in range(0, 7, 2):
  for polarity in ["POS", "NEG"]:
      name = "LM"+ polarity + str(i) + ".csv"
      key = polarity + str(i)
      print(name)
