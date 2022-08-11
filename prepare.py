import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from pydataset import data
import seaborn as sns
import math
import env
from sklearn.model_selection import train_test_split


def prep_iris(iris_df):
    iris_df.rename(columns={'species_name': 'species'}, inplace=True)
    dummy_iris_df = pd.get_dummies(iris_df[['species']])
    iris_df = pd.concat([iris_df, dummy_iris_df], axis=1)
    return iris_df

def split_data_iris(iris_df):
    train_iris, test_iris = train_test_split(iris_df, test_size=.2, random_state=123, stratify=iris_df.species)
    train_iris, validate_iris = train_test_split(train_iris, test_size=.25, random_state=123, stratify=train_iris.species)
    return train_iris, validate_iris, test_iris

def prep_titanic(titanic_df):
    titanic_df = titanic_df.drop_duplicates()
    cols_to_drop = ['deck', 'embarked', 'class', 'age']
    titanic_df = titanic_df.drop(columns=cols_to_drop)
    titanic_df['embark_town'] = titanic_df.embark_town.fillna(value='Southampton')
    dummy_titanic_df = pd.get_dummies(titanic_df[['sex','embark_town']], dummy_na=False, drop_first=[True, True])
    titanic_df = pd.concat([titanic_df, dummy_titanic_df], axis=1)
    return titanic_df

def split_data_titanic(titanic_df):
    train_titanic, test_titanic = train_test_split(titanic_df, test_size=.2, random_state=123, stratify=titanic_df.survived)
    train_titanic, validate_titanic = train_test_split(train_titanic, test_size=.25, random_state=123, stratify=train_titanic.survived)
    return train_titanic, validate_titanic, test_titanic

def prep_telco(telco_df):
    telco_df = telco_df.T.drop_duplicates().T
    dummy_telco_df = pd.get_dummies(telco_df[['gender','contract_type','internet_service_type']], dummy_na=False, drop_first=[True, False, False])
    telco_df = pd.concat([telco_df, dummy_telco_df], axis=1)
    telco_df.senior_citizen = telco_df.senior_citizen.astype('int')
    telco_df.tenure = telco_df.tenure.astype('int')
    telco_df.monthly_charges = telco_df.monthly_charges.astype('float')
    telco_df.partner = telco_df.partner.map(dict(Yes=1, No=0))
    telco_df.dependents = telco_df.dependents.map(dict(Yes=1, No=0))
    telco_df.phone_service = telco_df.phone_service.map(dict(Yes=1, No=0))
    telco_df.paperless_billing = telco_df.paperless_billing.map(dict(Yes=1, No=0))
    telco_df.churn = telco_df.churn.map(dict(Yes=1, No=0))
    cols_to_drop = ['internet_service_type_id', 'contract_type_id', 'payment_type_id', 'gender']
    telco_df = telco_df.drop(columns=cols_to_drop)
    telco_df.total_charges = telco_df.total_charges.str.replace(' ', '0')
    telco_df.total_charges = telco_df.total_charges.astype('float')
    return telco_df

def split_data_telco(telco_df):
    train_telco, test_telco = train_test_split(telco_df, test_size=.2, random_state=123, stratify=telco_df.churn)
    train_telco, validate_telco = train_test_split(train_telco, test_size=.25, random_state=123, stratify=train_telco.churn)
    return train_telco, validate_telco, test_telco


# train, test = train_test_split(df, test_size=.2, random_state=123, stratify=df.survived)
# train, validate = train_test_split(train, test_size=.3, random_state=123, stratify=train.survived)

# def split_data(df):
    '''
    take in a DataFrame and return train, validate, and test DataFrames; stratify on survived.
    return train, validate, test DataFrames.
    '''
    # train_validate, test = train_test_split(df, test_size=.2, random_state=123, stratify=df.survived)
    # train, validate = train_test_split(train_validate, 
                                    #    test_size=.3, 
                                    #    random_state=123, 
                                    #    stratify=train_validate.survived)
    # return train, validate, test