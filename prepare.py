import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from pydataset import data
import seaborn as sns
import math
import env

def prep_iris(iris_df):
    iris_df.rename(columns={'species_name': 'species'}, inplace=True)
    dummy_iris_df = pd.get_dummies(iris_df[['species']])
    iris_df = pd.concat([iris_df, dummy_iris_df], axis=1)
    return iris_df

def prep_titanic(titanic_df):
    titanic_df = titanic_df.drop_duplicates()
    cols_to_drop = ['deck', 'embarked', 'class', 'age']
    titanic_df = titanic_df.drop(columns=cols_to_drop)
    titanic_df['embark_town'] = titanic_df.embark_town.fillna(value='Southampton')
    dummy_titanic_df = pd.get_dummies(titanic_df[['sex','embark_town']], dummy_na=False, drop_first=[True, True])
    titanic_df = pd.concat([titanic_df, dummy_titanic_df], axis=1)
    return titanic_df

def prep_telco(telco_df):
    cols_to_drop = ['internet_service_type_id', 'contract_type_id', 'payment_type_id']
    telco_df = telco_df.drop(columns=cols_to_drop)
    telco_df = telco_df.T.drop_duplicates().T
    dummy_telco_df = pd.get_dummies(telco_df[['gender','contract_type','internet_service_type']], dummy_na=False, drop_first=[True, True, True])
    telco_df = pd.concat([telco_df, dummy_telco_df], axis=1)
    return telco_df