import pandas as pd
import os
# import sys
# sys.path.append('/Users/mph/codeup-data-science/classification-exercises/')
import env

def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def new_titanic_data():
    return pd.read_sql('SELECT * FROM passengers', get_connection('titanic_db'))

def get_titanic_data():
    filename = "titanic.csv"
    
    # if file is available locally, read it
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    
    # if file not available locally, acquire data from SQL database
    # and write it as csv locally for future use
    else:
        # read the SQL query into a dataframe
        df = new_titanic_data()
        
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename, index=False)

        # Return the dataframe to the calling code
        return df  

def new_iris_data():
    return pd.read_sql('SELECT * FROM species JOIN measurements USING(species_id);', get_connection('iris_db'))

def get_iris_data():
    filename2 = "iris.csv"
    
    # if file is available locally, read it
    if os.path.isfile(filename2):
        return pd.read_csv(filename2)
    
    # if file not available locally, acquire data from SQL database
    # and write it as csv locally for future use
    else:
        # read the SQL query into a dataframe
        df = new_iris_data()
        
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename2, index=False)

        # Return the dataframe to the calling code
        return df  

def new_telco_data():
    return pd.read_sql('SELECT * FROM customers JOIN contract_types USING(contract_type_id) JOIN customer_contracts USING(customer_id) JOIN customer_payments USING(customer_id) JOIN internet_service_types USING(internet_service_type_id);', get_connection('telco_churn'))

def get_telco_data():
    filename3 = "telco.csv"
    
    # if file is available locally, read it
    if os.path.isfile(filename3):
        return pd.read_csv(filename3)
    
    # if file not available locally, acquire data from SQL database
    # and write it as csv locally for future use
    else:
        # read the SQL query into a dataframe
        df = new_telco_data()
        
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename3, index=False)

        # Return the dataframe to the calling code
        return df  