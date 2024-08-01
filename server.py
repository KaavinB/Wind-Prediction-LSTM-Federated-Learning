import flwr as fl
from typing import List, Tuple
from flwr.common import Metrics

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    examples = [num_examples for num_examples, _ in metrics]
    total_examples = sum(examples)
    weighted_metrics = {}

    for _, m in metrics:
        for key, value in m.items():
            weighted_metrics[key] = weighted_metrics.get(key, 0) + value * (sum(examples) / total_examples)

    return weighted_metrics

def main():
    print("Server is starting...")
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_fn=None,
        on_fit_config_fn=None,
        on_evaluate_config_fn=None,
        initial_parameters=None,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=50),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()