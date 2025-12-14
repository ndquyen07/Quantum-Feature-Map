import numpy as np
import pickle
import time
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import sklearn here for main process
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from qiskit_algorithms.optimizers import COBYLA, SPSA, ADAM

# Import custom modules
from src.paramertrized_circuit import ParametrizedCircuit




def train_single_loss(loss_type, X_train, y_train, X_val, y_val, optimizer, init_theta, depth, type_ansatz, warmup_iterations, results_dir, maxiter, run_id):
    """
    Train a single TQFM with a specific loss function.
    This function runs in a separate process.
    """
    # Re-import modules in the subprocess (required for multiprocessing)
    import numpy as np
    import pickle
    import time
    import multiprocessing as mp

    print(f"Process {mp.current_process().name}: Starting training with {loss_type} loss")

    try:
        from src.feature_map import TrainableQuantumFeatureMap

        # Create TQFM with specific loss function
        tqfm = TrainableQuantumFeatureMap(
            depth=depth,
            type_ansatz=type_ansatz,
            type_loss=loss_type,
            warmup_iterations=warmup_iterations
        )

        # Train TQFM
        start_time = time.time()
        tqfm.fit(X_train, y_train, X_val, y_val, optimizer=optimizer, init_theta=init_theta.copy())
        training_time = time.time() - start_time

        print(f"Process {mp.current_process().name}: {loss_type} training completed in {training_time:.2f} seconds")

        # Save individual model  
        model_filename = f"{results_dir}/tqfm_{loss_type}_depth{depth}_ansatz{type_ansatz}_iter{maxiter}_run{run_id}.pkl"
        with open(model_filename, 'wb') as f:
            pickle.dump(tqfm, f)
        print(f"Process {mp.current_process().name}: Model saved to: {model_filename}")

        return {
            'loss_type': loss_type,
            'tqfm': tqfm,
            'training_time': training_time,
            'optimal_value': tqfm.optimal_value,
            'model_filename': model_filename
        }

    except Exception as e:
        print(f"Process {mp.current_process().name}: Error training {loss_type}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'loss_type': loss_type,
            'error': str(e),
            'training_time': 0,
            'optimal_value': None
        }


def main_parallel(run_id=0, depth=1, type_ansatz="RealAmplitudes", optimizer_name="COBYLA",
                maxiter=1000, warmup_iterations=200, n_samples=150, test_size=50):
    """Run TQFM training with three different loss functions in parallel."""

    # Validate parameters
    if depth < 1:
        raise ValueError("depth must be >= 1")
    if maxiter < 1:
        raise ValueError("maxiter must be >= 1")
    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be >= 0")
    if warmup_iterations >= maxiter:
        raise ValueError("warmup_iterations must be < maxiter")
    if type_ansatz not in ["TwoLocal", "RealAmplitudes", "EfficientSU2"]:
        raise ValueError(f"Invalid ansatz type: {type_ansatz}. Must be TwoLocal, RealAmplitudes, or EfficientSU2")
    if optimizer_name not in ["COBYLA", "SPSA", "ADAM"]:
        raise ValueError(f"Invalid optimizer: {optimizer_name}. Must be COBYLA, SPSA, or ADAM")
    if n_samples < test_size + 10:
        raise ValueError(f"n_samples ({n_samples}) must be at least test_size ({test_size}) + 10")

    # Check available CPU cores
    available_cores = mp.cpu_count()
    num_workers = min(len(["trace_distance", "hilbert_schmidt", "inner_loss"]), available_cores)
    print(f"Available CPU cores: {available_cores}, using {num_workers} workers")

    # Create results directory (different from sequential version)
    results_dir = "results_tqfm_losses_moon"
    os.makedirs(results_dir, exist_ok=True)

    # Generate data (same as in notebook)
    X, y = make_moons(n_samples=n_samples, noise=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)

    print(f"Data loaded - Training: {X_train.shape[0]} samples, Validation: {X_val.shape[0]} samples")

    # Display configuration
    print(f"\nConfiguration:")
    print(f"  Run ID: {run_id}")
    print(f"  Depth: {depth}")
    print(f"  Ansatz: {type_ansatz}")
    print(f"  Optimizer: {optimizer_name}")
    print(f"  Max iterations: {maxiter}")
    print(f"  Warmup iterations: {warmup_iterations}")
    print(f"  Training samples: {n_samples - test_size}")
    print(f"  Validation samples: {test_size}")

    # Create optimizer
    optimizer_map = {
        "COBYLA": COBYLA(maxiter=maxiter),
        "SPSA": SPSA(maxiter=maxiter, learning_rate=0.05, perturbation=0.01),
        "ADAM": ADAM(maxiter=maxiter, lr=0.05)
    }
    optimizer = optimizer_map[optimizer_name]

    # Create circuit based on ansatz type to get parameter count
    if type_ansatz == "TwoLocal":
        circuit = ParametrizedCircuit.TwoLocal_circuit(X_train.shape[1], depth)
    elif type_ansatz == "RealAmplitudes":
        circuit = ParametrizedCircuit.RealAmplitudes_circuit(X_train.shape[1], depth)
    elif type_ansatz == "EfficientSU2":
        circuit = ParametrizedCircuit.EfficientSU2_circuit(X_train.shape[1], depth)
    else:
        raise ValueError(f"Unknown ansatz type: {type_ansatz}")
    
    num_params = len(circuit.parameters) - X_train.shape[1]

    # Generate shared initial theta
    shared_init_theta = np.random.uniform(-np.pi, np.pi, num_params)
    print(f"Shared initial theta shape: {shared_init_theta.shape}")

    # Three loss functions to test
    loss_functions = ["trace_distance", "hilbert_schmidt", "inner_loss"]
    trained_tqfms = {}

    print("\n" + "="*80)
    print("Training TQFM with three loss functions in PARALLEL...")
    print("="*80)

    # Use ProcessPoolExecutor for parallel execution
    start_time_total = time.time()
    
    print(f"Starting parallel training with {num_workers} workers...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all training jobs
        future_to_loss = {
            executor.submit(
                train_single_loss,
                loss_type,
                X_train, y_train, X_val, y_val,
                optimizer,
                shared_init_theta,
                depth, type_ansatz, warmup_iterations,
                results_dir,
                maxiter,
                run_id
            ): loss_type for loss_type in loss_functions
        }

        # Collect results as they complete
        for future in as_completed(future_to_loss):
            loss_type = future_to_loss[future]
            try:
                result = future.result()
                if 'error' in result:
                    print(f"[FAIL] {loss_type} failed: {result['error']}")
                else:
                    trained_tqfms[loss_type] = result['tqfm']
                    print(f"[OK] {loss_type} completed: optimal_value={result['optimal_value']:.6f}, "
                          f"time={result['training_time']:.2f}s")

            except Exception as exc:
                print(f"[ERROR] {loss_type} generated an exception: {exc}")

    total_training_time = time.time() - start_time_total

    # Save shared initial theta
    init_theta_filename = f"{results_dir}/shared_init_theta_depth{depth}_ansatz{type_ansatz}_run{run_id}.npy"
    np.save(init_theta_filename, shared_init_theta)
    print(f"\nShared initial theta saved to: {init_theta_filename}")



    print("\n" + "="*80)
    print("PARALLEL TRAINING SUMMARY")
    print("="*80)
    print(f"Depth: {depth}")
    print(f"Ansatz: {type_ansatz}")
    print(f"Optimizer: {optimizer_name}")
    print(f"Max iterations: {maxiter}")
    print(f"Warmup iterations: {warmup_iterations}")
    print(f"Shared initial theta shape: {shared_init_theta.shape}")
    print(f"Total training time: {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)")

    for loss_type in loss_functions:
        if loss_type in trained_tqfms:
            print(f"{loss_type}: optimal_value = {trained_tqfms[loss_type].optimal_value:.6f}")
        else:
            print(f"{loss_type}: FAILED")

    print(f"\nAll results saved in: {results_dir}/")
    print("="*80)

    return trained_tqfms, shared_init_theta



if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run TQFM training with three loss functions in parallel')
    parser.add_argument("--run_id", type=int, default=0, help="Run ID for experiment tracking")
    parser.add_argument("--depth", type=int, default=1, help="Circuit depth")
    parser.add_argument("--ansatz", type=str, default="RealAmplitudes", help="Ansatz type")
    parser.add_argument("--optimizer", type=str, default="COBYLA", choices=["COBYLA", "SPSA", "ADAM"], help="Optimizer")
    parser.add_argument("--maxiter", type=int, default=1000, help="Maximum iterations")
    parser.add_argument("--warmup", type=int, default=200, help="Warmup iterations")
    parser.add_argument("--n_samples", type=int, default=150, help="Total number of samples")
    parser.add_argument("--test_size", type=int, default=50, help="Test set size")
    args = parser.parse_args()

    main_parallel(run_id=args.run_id, depth=args.depth, type_ansatz=args.ansatz,
                 optimizer_name=args.optimizer, maxiter=args.maxiter,
                 warmup_iterations=args.warmup, n_samples=args.n_samples,
                 test_size=args.test_size)