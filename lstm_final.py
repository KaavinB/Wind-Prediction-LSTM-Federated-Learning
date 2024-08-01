import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np
from sklearn.metrics import mean_squared_error

# Load data
df = pd.read_csv('jandata.csv')

# Data preprocessing
df.index = pd.to_datetime(df['Datetime'], format='%d-%m-%Y %H:%M')
df.drop(['Datetime', 'Resolution code', 'Decremental bid Indicator'], axis=1, inplace=True)

# Label Encoding
label_encoder = LabelEncoder()
df['Region'] = label_encoder.fit_transform(df['Region'])
df['Grid connection type'] = label_encoder.fit_transform(df['Grid connection type'])
df['Offshore/onshore'] = label_encoder.fit_transform(df['Offshore/onshore'])

# Feature engineering
df_filtered = df.copy()
columns = list(df_filtered.columns.values)

for feature in columns:
    max_threshold = df_filtered[feature].quantile(0.95)
    min_threshold = df_filtered[feature].quantile(0.05)
    df_filtered = df_filtered[(df_filtered[feature] <= max_threshold) & (df_filtered[feature] >= min_threshold)]

df_filtered = df_filtered[:2000]

# Split data into features and target
X = df_filtered.drop(['Most recent forecast'], axis=1)
Y = df_filtered['Most recent forecast']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=42, test_size=0.5)

# Reshape data for LSTM model
x_train_lstm = x_train.values.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test_lstm = x_test.values.reshape((x_test.shape[0], x_test.shape[1], 1))

# LSTM Model
model_lstm = Sequential([
    LSTM(64, input_shape=(x_train.shape[1], 1)),
    Dense(8, activation='relu'),
    Dense(1, activation='linear')
])

# Compile and fit the model
model_lstm.compile(optimizer='adam', loss='mean_absolute_percentage_error', metrics=['mse'])
model_lstm.fit(x_train_lstm, y_train, epochs=750, verbose=1)

# Evaluation
ypred_lstm = model_lstm.predict(x_test_lstm)
mse_lstm = mean_absolute_percentage_error(y_test, ypred_lstm)
print("Mean Absolute Percentage Error (LSTM):", mse_lstm)

r2 = r2_score(y_test, ypred_lstm)
print(f'R2 Score: {r2}')

# Plot predictions
def plot_predictions(model, X, y, start=0, end=100):
    predictions = model.predict(X).flatten()
    mse_value = mean_squared_error(y, predictions)

    df = pd.DataFrame(data={'Predictions': predictions, 'Actuals': y})

    plt.plot(df['Predictions'][start:end], label='Predictions')
    plt.plot(df['Actuals'][start:end], label='Actuals')

    plt.title('Actual vs Predicted Values')
    plt.xlabel('Sample Index')
    plt.ylabel('Values')
    plt.legend()

    plt.show()

    return df, mse_value

plot_predictions(model_lstm, x_train_lstm, y_train)

# Make a prediction
def prediction(test):
    test_data = np.array(test)
    test_data_lstm = test_data.reshape((test_data.shape[0], test_data.shape[1], 1))
    prediction = model_lstm.predict(test_data_lstm)
    print("Predicted Value:", prediction[0, 0])

test = [[1, 1, 1, 23.56, 9.23, 32.55, 28.92, 16.35, 40.93, 28.1, 15.59, 40.13, 21.98, 3.12, 40.56, 53.86, 0.44]]
prediction(test)

# Summary of the model
model_lstm.summary()
