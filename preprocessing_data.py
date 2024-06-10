import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def isna(data):
    return data.isna().sum()

def nan_finder(data):
    drops = []
    for name in data.columns:
        if data.isna().sum()[name] > (data.shape[0] * 0.4):
            drops.append(name)
    return drops

def nan_dropper(data,drops):
    data = data.drop(drops,axis=1)
    return data

def catches_columns_type(data,type:str,columns_list):
    for name in data.columns:
        if data[name].dtype  == type:
            columns_list.append(name)
    return columns_list

def useless_object_catcher(data):
    names = []
    for name in data.columns:
        if (data[name].value_counts().max() / data.shape[0]) > 0.9:
            data = data.drop(name,axis=1)
            names.append(name)
    print(f"Removed columns are \n {names}")
    return data.columns

def nan_object_filler(data):
    for name in data.columns:
        if data[name].isna().sum() > 0:
            most_repeated_value = data[name].value_counts().idxmax()
            data[name].fillna(most_repeated_value, inplace=True)
    return data

def mapping_object_values(data):
    le = LabelEncoder()
    for name in data.columns:
        data[name] = le.fit_transform(data[name])
    return data

def int_encoder(data):
    le = LabelEncoder()
    data = data.fit_transform(data)
    return data
    

    



    