# Wind Forecasting with Federated Learning

This repository is a small experiment in training an LSTM model with Flower. The example uses wind forecast data from `jandata.csv` and averages model updates on a server rather than combining raw records there.

## Run the federated example

Install the dependencies:

```bash
python -m pip install tensorflow pandas numpy scikit-learn flwr
```

Keep `jandata.csv` in the project directory. Start the server in one terminal:

```bash
python server.py
```

Start the client in two separate terminals:

```bash
python run_federated_learning.py
```

There is an important limitation in the current setup: the server is configured to wait for at least two clients, while each run of `run_federated_learning.py` starts one client. Starting that script twice will meet the server's client count, but both clients load and train on the same full dataset. The launcher does not currently assign separate data to each client, so this is not yet a useful simulation of independent data owners.

The server runs 50 rounds of FedAvg. Each client trains locally for five epochs per round and reports loss and mean squared error on its test split.

## Data and preprocessing

The CSV must include `Datetime`, `Resolution code`, `Decremental bid Indicator`, `Region`, `Grid connection type`, `Offshore/onshore`, and `Most recent forecast`. Other numeric columns are used as input features. The preprocessing code parses dates in `%d-%m-%Y %H:%M` format, removes the date and two metadata columns, label-encodes the three categorical columns, filters values outside the 5th to 95th percentile range, and keeps at most 2,000 rows.

`Most recent forecast` is the target. The rows are split randomly into training and test sets. Although the model uses an LSTM layer, the current code reshapes each row's features as the LSTM input; it does not build sliding windows of consecutive timestamps. The train/test split is not chronological.

## Files

- `prepare_data.py` loads, filters, and splits the CSV data.
- `lstm_model.py` defines the Keras model and parameter helpers.
- `client.py` implements the Flower client and its local training and evaluation.
- `run_federated_learning.py` loads the data and starts one client.
- `server.py` starts the Flower server and configures FedAvg.
- `lstm_final.py` contains a separate, non-federated training and plotting experiment.
