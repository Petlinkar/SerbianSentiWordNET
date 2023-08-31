#inferRNN
from srpskiwordnet import SrbWordNetReader
import pandas as pd
import os
import tensorflow as tf


def estimate_polarity(text, est_POS, est_NEG):
    a = est_POS.predict(text)
    b = est_NEG.predict(text)
    return (a * (1 - b), b * (1 - a))  

ROOT_DIR = ""
RES_DIR = os.path.join(ROOT_DIR, "resources")
MOD_DIR = os.path.join(ROOT_DIR, "ml_models")
POLARITY = ["POS", "NEG"]
ITERATION = ["0", "2", "4", "6"]

def main():
    swordnet = SrbWordNetReader(RES_DIR, "wnsrp30.xml") 

if __name__ == "__main__":
    main()