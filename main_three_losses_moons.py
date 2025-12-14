import numpy as np
import pickle
import time
import os
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from qiskit_algorithms.optimizers import COBYLA, SPSA, ADAM

from src.feature_map import TrainableQuantumFeatureMap
from src.paramertrized_circuit import ParametrizedCircuit



def main():
    """Run TQFM training with three different loss functions using the same initial theta."""

    # Create results directory
    results_dir = "results_tqfm_losses"
    os.makedirs(results_dir, exist_ok=True)

    # Generate data (same as in notebook)
    X, y = make_moons(n_samples=150, noise=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=50, random_state=42)

    print(f"Data loaded - Training: {X_train.shape[0]} samples, Validation: {X_val.shape[0]} samples")

    # Configuration
    depth = 1
    type_ansatz = "RealAmplitudes"
    optimizer_name = "COBYLA"
    maxiter = 1000
    warmup_iterations = 200

    # Create optimizer
    optimizer_map = {
        "COBYLA": COBYLA(maxiter=maxiter),
        "SPSA": SPSA(maxiter=maxiter, learning_rate=0.05, perturbation=0.01),
        "ADAM": ADAM(maxiter=maxiter, lr=0.05)
    }
    optimizer = optimizer_map[optimizer_name]


    # First create a dummy TQFM to get the circuit and parameter count
    circuit = ParametrizedCircuit.RealAmplitudes_circuit(X_train.shape[1], depth)
    num_params = len(circuit.parameters) - X_train.shape[1]

    # Generate shared initial theta
    shared_init_theta = np.random.uniform(-np.pi, np.pi, num_params)
    print(f"Shared initial theta shape: {shared_init_theta.shape}")
    print(f"Shared initial theta: {shared_init_theta}")

    # Three loss functions to test
    loss_functions = ["trace_distance", "hilbert_schmidt", "inner_loss"]
    trained_tqfms = {}

    print("\n" + "="*80)
    print("Training TQFM with three loss functions...")
    print("="*80)

    for loss_type in loss_functions:
        print(f"\n--- Training with {loss_type} loss ---")

        # Create TQFM with specific loss function
        tqfm = TrainableQuantumFeatureMap(
            depth=depth,
            type_ansatz=type_ansatz,
            type_loss=loss_type,
            warmup_iterations=warmup_iterations
        )

        # Train TQFM
        start_time = time.time()
        tqfm.fit(X_train, y_train, X_val, y_val, optimizer=optimizer, init_theta=shared_init_theta.copy())
        training_time = time.time() - start_time

        print(f"Optimal value: {tqfm.optimal_value:.6f}")
        print(f"Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

        # Store trained model
        trained_tqfms[loss_type] = tqfm


        # Save individual model
        model_filename = f"{results_dir}/tqfm_{loss_type}_depth{depth}_ansatz{type_ansatz}_iter{maxiter}.pkl"
        with open(model_filename, 'wb') as f:
            pickle.dump(tqfm, f)
        print(f"Model saved to: {model_filename}")



    # Save shared initial theta
    init_theta_filename = f"{results_dir}/shared_init_theta_depth{depth}_ansatz{type_ansatz}.npy"
    np.save(init_theta_filename, shared_init_theta)
    print(f"\nShared initial theta saved to: {init_theta_filename}")


    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Depth: {depth}")
    print(f"Ansatz: {type_ansatz}")
    print(f"Optimizer: {optimizer_name}")
    print(f"Max iterations: {maxiter}")
    print(f"Warmup iterations: {warmup_iterations}")
    print(f"Shared initial theta shape: {shared_init_theta.shape}")


    print(f"\nAll results saved in: {results_dir}/")
    print("="*80)

    return trained_tqfms, shared_init_theta


if __name__ == "__main__":
    main()