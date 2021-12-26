from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from datetime import date
import pandas as pd
import numpy as np
import string 
import pickle
import csv
import os

# GOOD/BAD payer
GOOD_BAD_PAYER = {1: 'KO - Bad payer', 0: 'OK - Good payer'}

# YES/NO dictionary
YES_NO_BINARY = {'NO': 0, 'YES': 1}

def dataset_generator(xlsx_file_path, csv_file_path, index_cols, flag):

    # Convert xlsx file to csv (if doesn't exist)
    if not os.path.isfile(csv_file_path):
        pd.read_excel(xlsx_file_path).to_csv(csv_file_path, columns=index_cols, index=False)
    
    # Generate data dictionary
    client_data = dict()
    with open(csv_file_path, 'r') as data:
        csv_reader = csv.reader(data)
        header = True
        for record in csv_reader:
            if header: header = False # Skip header
            else:
                if flag: client_data.setdefault(record[0], [record[i] for i in range(1, len(index_cols))])
                else:
                    # Check for ID_CLIENT data anomaly
                    if record[0].isnumeric():
                        if record[0] not in client_data.keys(): client_data.setdefault(record[0], [[record[i] for i in range(1, len(index_cols))]])
                        else: client_data[record[0]].append([record[i] for i in range(1, len(index_cols))])

    return client_data

def data_preparation(data, flag):

    new_dataset = dict()
    
    if flag:
        
        # ACTIVATION_CHANNEL dictionary
        ACTIVATION_CHANNEL = {'Web': 0, 'Pull': 1, 'Push': 2}
        for record in data.items():
            # Check if TARGET value is defined (1.0, 0.0), skip the record otherwise
            if record[1][1] in ['1.0', '0.0']:
                # Aviod KeyError for "strange" values (outlier) 
                try:
                    new_dataset.setdefault(record[0], [ACTIVATION_CHANNEL[record[1][0]], int(float(record[1][1])), YES_NO_BINARY[record[1][2]]])
                except KeyError:
                    new_dataset.setdefault(record[0], [3, int(float(record[1][1])), YES_NO_BINARY[record[1][2]]])  
    
    else:

        # Generate simple and derived features
        for key in data.keys():
            avg_day_delay_payment = 0
            total_day_of_service = 0
            bad_payment = 0
            
            for i in range(len(data.get(key))):
                avg_day_delay_payment += float(data.get(key)[i][1])
                total_day_of_service += (date(int(data.get(key)[i][3].split('-')[0]), int(data.get(key)[i][3].split('-')[1]), int(data.get(key)[i][3].split('-')[2])) - date(int(data.get(key)[i][2].split('-')[0]), int(data.get(key)[i][2].split('-')[1]), int(data.get(key)[i][2].split('-')[2]))).days
                bad_payment += YES_NO_BINARY[data.get(key)[i][4]]

            new_dataset.setdefault(key, [len(data.get(key)), round(avg_day_delay_payment / len(data.get(key))), total_day_of_service, int(bad_payment / len(data.get(key)) * 100)])
            
    return new_dataset

if __name__ == '__main__':

    # Check for if both a model and a dataset exists
    if os.path.isfile('model.pickle') and os.path.isfile('dataset.csv'):
        model = pickle.load(open('model.pickle', 'rb'))

    else:
            
        # Create dataset (if doesn't exist)
        if not os.path.isfile('dataset.csv'):

            # Convert xlsx to csv
            client_personal_data = dataset_generator('dataset/Client_Personal_Data.xlsx', 'dataset/client_personal_data.csv', ['ID_CLIENT', 'ACTIVATION_CHANNEL', 'TARGET', 'FLG_DOM_BANK'], 1)
            client_history = dataset_generator('dataset/Client_History.xlsx', 'dataset/client_history.csv', ['ID_CLIENT', 'SERVICE_ID', 'DAY_DELAY_PAYMENT', 'SERVICE_START_DATE', 'SERVICE_END_DATE', 'FLAG_BAD_CLIENT'], 0)

            # Data preparation
            client_personal_data_checked = data_preparation(client_personal_data, 1)
            client_history_checked = data_preparation(client_history, 0)

            # Dataset generation
            with open('dataset.csv', 'w') as data:
                writer = csv.writer(data)

                # Write header
                writer.writerow(['ACTIVATION_CHANNEL',
                                 'TARGET',
                                 'FLG_DOM_BANK',
                                 'SERVICES_NUMBER', 
                                 'AVG_DAY_DELAY_PAYMENT', 
                                 'DAY_OF_ACTIVE_SERVICES',
                                 'BAD_PAYMENT(%)'])
                
                for key in client_personal_data_checked.keys():

                    # Write record
                    writer.writerow([client_personal_data_checked.get(key)[0],
                                     client_personal_data_checked.get(key)[1],
                                     client_personal_data_checked.get(key)[2],
                                     client_history_checked.get(key)[0],
                                     client_history_checked.get(key)[1],
                                     client_history_checked.get(key)[2],
                                     client_history_checked.get(key)[3]])
        
        # Create model (if doesn't exist)
        if not os.path.isfile('model.pickle'):
            
            # Initialize the model pipeline
            model = make_pipeline(StandardScaler(), LogisticRegression()) 

            # Save the model (if doesn't exist)
            with open('model.pickle', 'wb') as fl:
                pickle.dump(model, fl)
        
    # Read dataset with Pandas
    dataset = pd.read_csv('dataset.csv')

    # Split the dataset in train and test
    x_train, x_test, y_train, y_test = train_test_split(dataset.drop('TARGET', axis=1), dataset['TARGET']) 
    
    # Train the model
    model.fit(x_train.values, y_train.values)

    # Prediction sample
    print('\nPREDICTION SAMPLES')
    array = np.array([2,1,4,5,229,100])
    print('Predicted: {} - Expected: KO - Bad payer'.format(GOOD_BAD_PAYER[model.predict(array.reshape(1, 6))[0]])) 
    array = np.array([2,1,3,1,279,0])
    print('Predicted: {} - Expected: OK - Good payer'.format(GOOD_BAD_PAYER[model.predict(array.reshape(1, 6))[0]]))

    # Calculate scores
    print('\nMODEL METRICS')
    print('Accuracy score: ', round(accuracy_score(y_test, model.predict(x_test.values)), 5))
    print('Recall score:   ', round(recall_score(y_test, model.predict(x_test.values)), 5), '\n')
