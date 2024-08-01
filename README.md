# Federated Learning LSTM Model for Time Series Forecasting

This project implements a federated learning approach using LSTM (Long Short-Term Memory) neural networks for time series forecasting. The system is built using TensorFlow and Flower (flwr) for federated learning.

## Project Overview

This federated learning system allows multiple clients to collaboratively train an LSTM model without sharing their raw data. The model is designed to forecast time series data, specifically tailored for energy-related predictions.

### Key Components:

1. LSTM Model: Defined in `lstm_model.py`
2. Federated Learning Client: Implemented in `client.py`
3. Data Preparation: Handled by `prepare_data.py`
4. Federated Learning Server: Set up in `server.py`
5. Main Execution Script: `run_federated_learning.py`

## Setup and Installation

### 1. Clone this repository:
```bash
git clone https://github.com/KaavinB/Wind-Prediction-LSTM-Federated-Learning.git
 ```
### 2. Install dependencies:
```bash
pip install tensorflow pandas numpy scikit-learn flwr
```
3. Ensure `jandata.csv` is in the project directory.
4. Start the server: python server.py
5. Run clients in separate terminals: python run_federated_learning.py

## 🏗️ Project Structure

- `client.py`: Federated learning client (Flower's `NumPyClient`)
- `lstm_model.py`: LSTM model architecture and utilities
- `prepare_data.py`: Data loading, preprocessing, and splitting
- `run_federated_learning.py`: Main script to start a client
- `server.py`: Federated learning server setup and execution

## 📊 Data

Use `jandata.csv` with columns:
- Datetime
- Region
- Grid connection type
- Offshore/onshore
- Most recent forecast (target variable)
- Other relevant features

## 🛠️ Customization

- Adjust client numbers, LSTM architecture, or training parameters in respective files
- Modify data preprocessing in `prepare_data.py` for different datasets

## 🤝 Contributing

Contributions welcome! Fork the repo and submit a pull request with your changes.

## 🙏 Acknowledgments

Built with [Flower](https://flower.dev/) - a friendly federated learning framework.

---

<p align="center">
Made with ❤️ by Kaavin
</p>
