import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

def train_val_test(df, col):
    seed = 42 
    train, val_test = train_test_split(df, train_size=.7, random_state=seed, stratify=df[col])
    
    validate, test = train_test_split(val_test, train_size=.5, random_state=seed, stratify=val_test[col])
    
    return train, validate, test