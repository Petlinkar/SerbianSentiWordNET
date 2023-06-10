# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 10:47:59 2022.

@author: "Petalinkar Saša"

Main script for 

Engleske definicije bez ?
ENG30-01464700-a
ENG30-02295867-a
ENG30-02295867-a
ENG30-01825125-v
"""
from SerbSynpretproceing import showGrid, showGridReg
from SerbainTagger import SrbTreeTagger
from srpskiwordnet import SrbWordNetReader
from srppolsets import PolaritySets, syn2gloss
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from StemmerByNikola import stem_str
import numpy as np
from sklearn.neural_network import MLPClassifier
def load_file_into_list(filename):
    with open(filename, mode="r", encoding="utf-16") as file:
        lines = [line.strip() for line in file]
    return lines
from sklearn.naive_bayes import ComplementNB, BernoulliNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split

C_range = np.logspace(4, 13, 4)
gamma_range = np.logspace(-9, -5, 5)

alpha_range = np.logspace(-10, -7, 4)

# stop_words = [".", "i", "ili", "(", ")", ";", ",", "u", "iz", "se", "koji",
#               "na", "kao", "sa", "kojim", "koj"]
RES_DIR = ".\\resources\\"
MOD_DIR = ".\\ml_models\\"
swordnet = SrbWordNetReader("F:\Serbian Corpora\wnsrp30","wnsrp30.xml")
stop_words1 = load_file_into_list(RES_DIR + "stopwordsSRB.txt")
stop_words2 = load_file_into_list(RES_DIR + "stopwordsSRB tfidf.txt")
stop_words3 = load_file_into_list(RES_DIR + "stopwordsSRB tf.txt")
pos_df = pd.read_csv(RES_DIR + "pos.csv" )
neg_df = pd.read_csv(RES_DIR + "neg.csv" )
tt = SrbTreeTagger()
stop_words4 = list()
for word in stop_words3:
    pom = tt.lemmarizer(word)
    if not pom in stop_words4:
        stop_words4.append(pom)
# swordnet.parse_all_defintions(tt.lemmarizer)
Synset_Sentiment_set = PolaritySets(swordnet, 0)
stop_words3 = stop_words3 + ['daljem', 'izgledalo', 'izgledu', 'međuvremenu',
                             'sc', 'sl', 'slučaju', 'tekstu', 'vreme']

# Synset_Sentiment_set.addWNop()
# print("OBJ from WN-OP")

Synset_Sentiment_set.addWSWN()
print("OBJ from SWN")


positive_seed_words = ["dobar",
                       "dobrota",
                       "lep",
                       "čudesno",
                       "dragocen",
                       "anđeoski",
                       "izobilje",
                       "izbavljenje",
                       "tešiti",
                       "ispravnost",
                       "oduševiti se",
                       "slast",
                       "uveseljavajući",
                       "napredovati",
                       "proslavljen",
                       "usrećiti",
                       "uspešnost"]

negative_seed_words = ["zao",
                       "antipatija",
                       "beda",
                       "bedan",
                       "bol",
                       "laž",
                       "lažno",
                       "korupcija",
                       "krivica",
                       "prezreti",
                       "tuga",
                       "nauditi",
                       "sebičnost",
                       "paćeništvo",
                       "ukloniti s ovog sveta",
                       "masakr",
                       "ratovanje"]
positive_seed_IDS = ["ENG30-01828736-v",
                     "ENG30-13987905-n",
                     "ENG30-01777210-v",
                     "ENG30-13987423-n",
                     "ENG30-01128193-v",
                     "ENG30-02355596-v",
                     "ENG30-00271946-v",
                    ]

negative_seed_IDS = ["ENG30-01510444-a",
                     "ENG30-01327301-v",
                     "ENG30-00735936-n",
                     "ENG30-00220956-a",
                     "ENG30-02463582-a",
                     "ENG30-01230387-a",
                     "ENG30-00193480-a",
                     "ENG30-00364881-a",
                     "ENG30-14213199-n",
                     "ENG30-01792567-v",
                     "ENG30-07427060-n",
                     "ENG30-14408086-n",
                     "ENG30-14365741-n",
                     "ENG30-02466111-a",
                     "ENG30-14204950-n",
                     "ENG30-10609960-n",
                     "ENG30-02319050-v",
                     "ENG30-02495038-v",
                     "ENG30-01153548-n",
                     "ENG30-00751944-n",
                    ]

Synset_Sentiment_set.addPOSall(positive_seed_words)
Synset_Sentiment_set.addNEGall(negative_seed_words)
Synset_Sentiment_set.addPOSIDall(positive_seed_IDS)
Synset_Sentiment_set.addNEGIDall(negative_seed_IDS)
Synset_Sentiment_set.addPOSIDall(pos_df["ID"])
Synset_Sentiment_set.addNEGIDall(neg_df["ID"])
print("POS and NEG manuely chosen")
rem_obj = ["ENG30-05528604-n",
"ENG30-00749767-n",
"ENG30-09593651-n",
"ENG30-13250542-n",
"ENG30-13132338-n",
"ENG30-05943066-n",
"ENG30-03123143-a",
"ENG30-10104209-n",
"ENG30-12586298-n",
"ENG30-01971094-n",
"ENG30-00759269-v",
"ENG30-00948206-n",
"ENG30-01039307-n",
"ENG30-02041877-v",
"ENG30-00023271-n",
"ENG30-13509196-n",
"ENG30-09450866-n",
"ENG30-03947798-n",
"ENG30-08589140-n",
"ENG30-09569709-n",
"ENG30-00223268-n",
"ENG30-00220409-n",
"ENG30-00224936-n",
"ENG30-00222248-n",
"ENG30-00221981-n",
"ENG30-00223362-n",
"ENG30-00222485-n",
"ENG30-00223720-n",
"ENG30-00225593-n",
"ENG30-00221596-n",
"ENG30-00223268-n",
"ENG30-02485451-v",
"ENG30-02574205-v"

          ]
Synset_Sentiment_set.removeSynIDs(rem_obj)

# Synset_Sentiment_set.addWNopAll()
# print("POS, NEG and OBJ from WN-OP")
Synset_Sentiment_set.updateDataFrame()

Synset_Sentiment_sets = list()
Synset_Sentiment_sets.append(Synset_Sentiment_set)

for _ in range(6):
    Synset_Sentiment_sets.append(
        Synset_Sentiment_sets[-1].next_itteration())

pipeline = Pipeline(
    [
        # ("gloss", SrbSynset2GlossTransformer()),
        ("vect", HashingVectorizer()),
        ("tfidf", TfidfTransformer()),
        # ("minmax", MaxAbsScaler()),

        # ("svm", SVC()),
        # ("svm", SVR()),
        # ("nn", MLPClassifier())
        ("sgd", SGDClassifier())
        # ("nb", BernoulliNB())
        # ("tree", RandomForestClassifier())
        # ("ada", AdaBoostClassifier())
        
    ],
    memory="F:\\Temp\\ML_Cache"
)
pipeline
parameters = {
    # "vect__preprocessor": (tt.lemmarizer,),
    # "vect__max_df": (0.5, 0.75, 1.0),
    # "vect__stop_words": (stop_words1, stop_words2, stop_words3),
    "vect__stop_words": (stop_words3,),
    # 'vect__max_features': (5000, 10000, 50000),
    "vect__ngram_range": ((1, 1),),  # unigrams or bigrams
    'tfidf__use_idf': (True, ),
    # 'tfidf__norm': ('l1', 'l2'),
    # 'tfidf__norm': ('l1', ),
    # "svm__kernel": ('poly','sigmoid', 'rbf'),
    # "svm__probability": (True,),
    # "svm__class_weight": ("balanced", ),
    # "svm__C": C_range,
    # "svm__gamma": gamma_range,
    # "svm__C": (0.1, 1, 10, 1000),
    # "svm__gamma": (0.1, 1),

    # "svm__cache_size": (1000,)
    'sgd__loss': ('modified_huber', 'log_loss'),
    'sgd__penalty': ('l2', 'l1', 'elasticnet'),
    "sgd__class_weight": ("balanced", ),
    # "nn__alpha": (1, 5),
    # "nn__hidden_layer_sizes": ((100, 100, 100), (500, 250, 100, 10 ))
    # 'nb__alpha': (0.00001, 0.0001, 0.001, 0.1, 1, 10, 100, 1000),
    # 'nb__alpha': alpha_range,
    # 'nb__binarize': (0.0, 0.0001, 0.001)
    # "nb__norm": (True, False),
    # "tree__max_depth": (None, 2, 5, 10, 25),
    # "tree__class_weight": ("balanced", ),
    # "ada__n_estimators" :(100, 200, 500, 1000, 5000)
    
    }

    
    
# parameters
# defi = list()

# for s in Synset_Sentiment_sets:
#     defi.append(s.getDef())
pretprocess = [syn2gloss, tt.lemmarizer]
model = "SDG-klas-"
TRAIN_DIR = ".//train_sets//"
for i, pol_sets in enumerate(Synset_Sentiment_sets):
    for polarity in ["POS", "NEG"]:
        name = "LM"+ polarity + str(i) + ".csv"
        X, y = pol_sets.getXY(polarity, preprocess=pretprocess)
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                            random_state=13)
        X_train.to_csv(TRAIN_DIR+"X_train_" + name)
        y_train.to_csv(TRAIN_DIR+"y_train_" + name)
        X_test.to_csv(TRAIN_DIR+"X_test_" + name)
        y_test.to_csv(TRAIN_DIR+"y_test_" + name)

for i, pol_sets in enumerate(Synset_Sentiment_sets):
    for polarity in ["POS", "NEG"]:
        name = "UP"+ polarity + str(i) + ".csv"
        X, y = pol_sets.getXY(polarity)
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                            random_state=13)
        X_train.to_csv(TRAIN_DIR+"X_train_" + name)
        y_train.to_csv(TRAIN_DIR+"y_train_" + name)
        X_test.to_csv(TRAIN_DIR+"X_test_" + name)
        y_test.to_csv(TRAIN_DIR+"y_test_" + name)

        
# for i, s in enumerate(Synset_Sentiment_sets):
#     if (i % 2) != 0:
#         continue
#     # Training positive
#     X, y = s.getXY("POS", predprocess=pretprocess)
#     title = "Positive Iteration " + str(i)
#     print(title)
#     elstimator = showGrid(X, y, pipeline, parameters, title).best_estimator_
#     # elstimator = showGridReg(X, y, pipeline, parameters, title).best_estimator_

#     filename = model + "POS-" + str(i) + ".joblit"
#     dump(elstimator,MOD_DIR+filename)
#     #training negative
#     X, y = s.getXY("NEG", predprocess=pretprocess)
#     title = "Negative Iteration " + str(i)
#     print(title)
#     elstimator = showGrid(X, y, pipeline, parameters, title).best_estimator_
#     # elstimator = showGridReg(X, y, pipeline, parameters, title).best_estimator_
#     filename = model + "NEG-" + str(i) + ".joblit"
#     dump(elstimator,MOD_DIR+filename)
# print ("kfold is 5")

