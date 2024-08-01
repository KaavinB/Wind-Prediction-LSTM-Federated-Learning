import flwr as fl
from prepare_data import load_and_preprocess_data, split_data_for_clients
from client import client_fn

def main():
    print("Loading and preprocessing data...")
    x_train, y_train, x_test, y_test = load_and_preprocess_data('jandata.csv')
    
    print("Splitting data for clients...")
    client_data = split_data_for_clients(x_train, y_train, x_test, y_test, num_clients=1)

    print("Starting Flower client...")
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=client_fn(*client_data[0])
    )

if __name__ == "__main__":
    main()