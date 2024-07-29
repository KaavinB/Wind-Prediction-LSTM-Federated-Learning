import flwr as fl
import tensorflow as tf
from lstm_model import create_lstm_model, get_model_parameters, set_model_params

class LSTMClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def get_parameters(self, config):
        return get_model_parameters(self.model)

    def fit(self, parameters, config):
        set_model_params(self.model, parameters)
        self.model.fit(self.x_train, self.y_train, epochs=5, verbose=0)
        return get_model_parameters(self.model), len(self.x_train), {}

    def evaluate(self, parameters, config):
        set_model_params(self.model, parameters)
        loss, mse = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return loss, len(self.x_test), {"mse": mse}

def client_fn(x_train, y_train, x_test, y_test):
    model = create_lstm_model((x_train.shape[1], 1))
    return LSTMClient(model, x_train, y_train, x_test, y_test)