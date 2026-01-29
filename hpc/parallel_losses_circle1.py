"""
Parallel training of TQFM with multiple loss functions.
Refactored for better maintainability and efficiency.
"""
import numpy as np
import pickle
import time
import os
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from qiskit_algorithms.optimizers import COBYLA, SPSA, ADAM

from src.paramertrized_circuit import ParametrizedCircuit


# Configuration constants
LOSS_FUNCTIONS = ["pre_determine_inner", "hilbert_schmidt", "log_likelihood_macro"]
VALID_ANSATZ = ["Universal", "RealAmplitudes", "EfficientSU2"]
VALID_OPTIMIZERS = ["COBYLA", "SPSA", "ADAM"]


@dataclass
class TrainingConfig:
    """Configuration for TQFM training."""
    run_id: int = 0
    depth: int = 1
    type_ansatz: str = "RealAmplitudes"
    optimizer_name: str = "COBYLA"
    maxiter: int = 1000
    warmup_iterations: int = 200
    n_samples: int = 150
    test_size: int = 50
    results_dir: str = "/data/ndquyen/qfm/results_losses_circle"
    
    def validate(self):
        """Validate configuration parameters."""
        checks = [
            (self.depth >= 1, "depth must be >= 1"),
            (self.maxiter >= 1, "maxiter must be >= 1"),
            (self.warmup_iterations >= 0, "warmup_iterations must be >= 0"),
            (self.warmup_iterations < self.maxiter, "warmup_iterations must be < maxiter"),
            (self.type_ansatz in VALID_ANSATZ, f"ansatz must be one of {VALID_ANSATZ}"),
            (self.optimizer_name in VALID_OPTIMIZERS, f"optimizer must be one of {VALID_OPTIMIZERS}"),
            (self.n_samples >= self.test_size + 10, f"n_samples must be >= test_size + 10"),
        ]
        
        for condition, message in checks:
            if not condition:
                raise ValueError(message)


def create_optimizer(name: str, maxiter: int):
    """Factory function for creating optimizers."""
    optimizers = {
        "COBYLA": lambda: COBYLA(maxiter=maxiter),
        "SPSA": lambda: SPSA(maxiter=maxiter, learning_rate=0.05, perturbation=0.01),
        "ADAM": lambda: ADAM(maxiter=maxiter, lr=0.05)
    }
    return optimizers[name]()


def create_circuit(ansatz_type: str, num_qubits: int, depth: int):
    """Factory function for creating quantum circuits."""
    circuits = {
        "Universal": ParametrizedCircuit.Universal_circuit,
        "RealAmplitudes": ParametrizedCircuit.RealAmplitudes_circuit,
        "EfficientSU2": ParametrizedCircuit.EfficientSU2_circuit
    }
    return circuits[ansatz_type](num_qubits, depth)


def train_single_loss(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train a single TQFM with specific loss function.
    Accepts dict to simplify multiprocessing parameter passing.
    
    Args:
        args: Dictionary containing all training parameters
        
    Returns:
        Dictionary with training results
    """
    # Import in subprocess (required for multiprocessing on Windows)
    from src.feature_map_1 import TrainableQuantumFeatureMap
    import multiprocessing as mp
    
    loss_type = args['loss_type']
    process_name = mp.current_process().name
    
    print(f"[{process_name}] Starting {loss_type} training")

    try:
        # Create TQFM
        tqfm = TrainableQuantumFeatureMap(
            depth=args['depth'],
            type_ansatz=args['type_ansatz'],
            type_loss=loss_type,
            warmup_iterations=args['warmup_iterations']
        )

        # Train
        start_time = time.time()
        optimizer = create_optimizer(args['optimizer_name'], args['maxiter'])
        tqfm.fit(
            args['X_train'], args['y_train'], 
            args['X_val'], args['y_val'],
            optimizer=optimizer,
            init_theta=args['init_theta'].copy()
        )
        training_time = time.time() - start_time

        print(f"[{process_name}] {loss_type} completed in {training_time:.2f}s")

        # Save model
        model_filename = (
            f"{args['results_dir']}/tqfm_{loss_type}_"
            f"depth{args['depth']}_ansatz{args['type_ansatz']}_"
            f"iter{args['maxiter']}_run{args['run_id']}.pkl"
        )
        with open(model_filename, 'wb') as f:
            pickle.dump(tqfm, f)
        print(f"[{process_name}] Saved to {model_filename}")

        return {
            'loss_type': loss_type,
            'tqfm': tqfm,
            'training_time': training_time,
            'optimal_value': tqfm.optimal_value,
            'model_filename': model_filename,
            'success': True
        }

    except Exception as e:
        print(f"[{process_name}] Error in {loss_type}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'loss_type': loss_type,
            'error': str(e),
            'success': False
        }


def main_parallel(config: Optional[TrainingConfig] = None, **kwargs) -> tuple:
    """
    Run TQFM training with multiple loss functions in parallel.
    
    Args:
        config: TrainingConfig object, or pass parameters as kwargs
        
    Returns:
        Tuple of (trained_models_dict, shared_init_theta)
    """
    # Create config from kwargs if not provided
    if config is None:
        config = TrainingConfig(**kwargs)
    
    config.validate()
    
    # Setup
    os.makedirs(config.results_dir, exist_ok=True)
    
    # Generate data
    X, y = make_circles(n_samples=config.n_samples, noise=0.1, factor=0.5, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=config.test_size, random_state=42
    )



    # Create circuit to determine parameter count
    circuit = create_circuit(config.type_ansatz, X_train.shape[1], config.depth)
    num_params = len(circuit.parameters) - X_train.shape[1]
    shared_init_theta = np.random.uniform(-np.pi, np.pi, num_params)

    # Print configuration
    print(f"\n{'='*80}")
    print("PARALLEL TRAINING CONFIGURATION")
    print(f"{'='*80}")
    print(f"Run ID:              {config.run_id}")
    print(f"Depth:               {config.depth}")
    print(f"Ansatz:              {config.type_ansatz}")
    print(f"Optimizer:           {config.optimizer_name}")
    print(f"Max iterations:      {config.maxiter}")
    print(f"Warmup iterations:   {config.warmup_iterations}")
    print(f"Training samples:    {len(X_train)}")
    print(f"Validation samples:  {len(X_val)}")
    print(f"Loss functions:      {len(LOSS_FUNCTIONS)}")
    print(f"{'='*80}\n")

    # Prepare arguments for each loss function
    base_args = {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'init_theta': shared_init_theta,
        'depth': config.depth,
        'type_ansatz': config.type_ansatz,
        'optimizer_name': config.optimizer_name,
        'maxiter': config.maxiter,
        'warmup_iterations': config.warmup_iterations,
        'results_dir': config.results_dir,
        'run_id': config.run_id
    }
    
    training_args = [
        {**base_args, 'loss_type': loss} for loss in LOSS_FUNCTIONS
    ]

    # Parallel training
    trained_models = {}
    training_times = {}
    start_time = time.time()
    
    import multiprocessing as mp
    num_workers = min(len(LOSS_FUNCTIONS), mp.cpu_count())
    print(f"Starting parallel training with {num_workers} workers...\n")
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all jobs
        future_to_loss = {
            executor.submit(train_single_loss, args): args['loss_type']
            for args in training_args
        }

        # Collect results
        for future in as_completed(future_to_loss):
            loss_type = future_to_loss[future]
            try:
                result = future.result()
                if result['success']:
                    trained_models[loss_type] = result['tqfm']
                    training_times[loss_type] = result['training_time']
                    print(f"✓ {loss_type:25} | optimal={result['optimal_value']:.6f} | "
                          f"time={result['training_time']:.2f}s")
                else:
                    print(f"✗ {loss_type:25} | FAILED: {result.get('error', 'Unknown')}")
            except Exception as exc:
                print(f"✗ {loss_type:25} | EXCEPTION: {exc}")

    total_time = time.time() - start_time

    # Save shared initial theta
    init_filename = (
        f"{config.results_dir}/shared_init_theta_"
        f"depth{config.depth}_ansatz{config.type_ansatz}_run{config.run_id}.npy"
    )
    np.save(init_filename, shared_init_theta)

    # Save training times and metadata
    timing_data = {
        'total_parallel_time': total_time,
        'individual_training_times': training_times,
        'configuration': {
            'run_id': config.run_id,
            'depth': config.depth,
            'type_ansatz': config.type_ansatz,
            'optimizer_name': config.optimizer_name,
            'maxiter': config.maxiter,
            'warmup_iterations': config.warmup_iterations,
            'n_samples': config.n_samples,
            'test_size': config.test_size
        },
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    timing_filename = (
        f"{config.results_dir}/training_times_"
        f"depth{config.depth}_ansatz{config.type_ansatz}_run{config.run_id}.json"
    )
    with open(timing_filename, 'w') as f:
        json.dump(timing_data, f, indent=2)
    
    print(f"Training times saved to: {timing_filename}")

    # Summary
    print(f"\n{'='*80}")
    print("TRAINING SUMMARY")
    print(f"{'='*80}")
    print(f"Total time:          {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"Successful:          {len(trained_models)}/{len(LOSS_FUNCTIONS)}")
    print(f"Results saved in:    {config.results_dir}/")
    print(f"{'='*80}\n")

    return trained_models, shared_init_theta


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Parallel TQFM training with multiple loss functions'
    )
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--ansatz", type=str, default="RealAmplitudes")
    parser.add_argument("--optimizer", type=str, default="COBYLA")
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--n_samples", type=int, default=150)
    parser.add_argument("--test_size", type=int, default=50)
    
    args = parser.parse_args()

    config = TrainingConfig(
        run_id=args.run_id,
        depth=args.depth,
        type_ansatz=args.ansatz,
        optimizer_name=args.optimizer,
        maxiter=args.maxiter,
        warmup_iterations=args.warmup,
        n_samples=args.n_samples,
        test_size=args.test_size
    )

    main_parallel(config)
