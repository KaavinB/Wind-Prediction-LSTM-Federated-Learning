import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path, num_samples=2000):
    df = pd.read_csv(file_path)
    
    df.index = pd.to_datetime(df['Datetime'], format='%d-%m-%Y %H:%M')
    df.drop(['Datetime', 'Resolution code', 'Decremental bid Indicator'], axis=1, inplace=True)

    label_encoder = LabelEncoder()
    df['Region'] = label_encoder.fit_transform(df['Region'])
    df['Grid connection type'] = label_encoder.fit_transform(df['Grid connection type'])
    df['Offshore/onshore'] = label_encoder.fit_transform(df['Offshore/onshore'])

    df_filtered = df.copy()
    columns = list(df_filtered.columns.values)

    for feature in columns:
        max_threshold = df_filtered[feature].quantile(0.95)
        min_threshold = df_filtered[feature].quantile(0.05)
        df_filtered = df_filtered[(df_filtered[feature] <= max_threshold) & (df_filtered[feature] >= min_threshold)]

    df_filtered = df_filtered[:num_samples]

    X = df_filtered.drop(['Most recent forecast'], axis=1)
    Y = df_filtered['Most recent forecast']

    x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=42, test_size=0.5)

    x_train_lstm = x_train.values.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test_lstm = x_test.values.reshape((x_test.shape[0], x_test.shape[1], 1))

    return x_train_lstm, y_train.values, x_test_lstm, y_test.values

def split_data_for_clients(x_train, y_train, x_test, y_test, num_clients=2):
    train_splits = np.array_split(range(len(x_train)), num_clients)
    test_splits = np.array_split(range(len(x_test)), num_clients)
    
    return [
        (x_train[train_splits[i]], y_train[train_splits[i]], 
         x_test[test_splits[i]], y_test[test_splits[i]])
        for i in range(num_clients)
    ]