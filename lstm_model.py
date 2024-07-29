import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, input_shape=input_shape),
        Dense(8, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mean_absolute_percentage_error', metrics=['mse'])
    return model

def get_model_parameters(model):
    return model.get_weights()

def set_model_params(model, params):
    model.set_weights(params)