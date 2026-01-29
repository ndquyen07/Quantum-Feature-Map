import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix
from scipy.linalg import sqrtm
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional

# Constants
FIDELITY_CLIP_MIN = 1e-7
FIDELITY_CLIP_MAX = 1.0


class LossStrategy(ABC):
    """Base class for loss computation strategies."""
    
    @staticmethod
    def prepare_targets(labels: np.ndarray, class_patterns: Optional[Dict], 
                       num_qubits: int) -> Optional[Dict]:
        """Prepare target states if needed for the loss computation."""
        return None
    
    @staticmethod
    @abstractmethod
    def compute(rho_list: List[np.ndarray], num_classes: int, **kwargs) -> float:
        """Compute the loss value."""
        raise NotImplementedError


class TraceDistanceLoss(LossStrategy):
    """Trace distance loss between class density matrices."""
    
    @staticmethod
    def compute(rho_list: List[np.ndarray], num_classes: int, **kwargs) -> float:
        loss = sum(
            np.sum(np.abs(np.linalg.eigvalsh(rho_list[i] - rho_list[j])))
            for i in range(num_classes - 1)
            for j in range(i + 1, num_classes)
        )
        num_pairs = num_classes * (num_classes - 1)
        return 1 - loss / num_pairs if num_pairs > 0 else 0.0


class HilbertSchmidtLoss(LossStrategy):
    """Hilbert-Schmidt distance loss between class density matrices."""
    
    @staticmethod
    def compute(rho_list: List[np.ndarray], num_classes: int, **kwargs) -> float:
        loss = sum(
            np.trace((rho_list[i] - rho_list[j]) @ (rho_list[i] - rho_list[j]))
            for i in range(num_classes - 1)
            for j in range(i + 1, num_classes)
        )
        num_pairs = num_classes * (num_classes - 1)
        return 1 - 0.5 * loss / num_pairs if num_pairs > 0 else 0.0


class InnerLoss(LossStrategy):
    """Inner product loss with target basis states."""
    
    @staticmethod
    def prepare_targets(labels: np.ndarray, class_patterns: Optional[Dict], 
                       num_qubits: int) -> Dict:
        if class_patterns is None:
            raise ValueError("class_patterns must be set for inner_loss")
        return {
            label: Statevector.from_label(class_patterns[label])
            for label in labels
        }
    
    @staticmethod
    def compute(rho_list: List[np.ndarray], num_classes: int, 
                inner_products: List[np.ndarray], **kwargs) -> float:
        """Compute loss as 1 - mean(class_average_fidelities)."""
        class_means = [np.mean(fids) for fids in inner_products if len(fids) > 0]
        return 1.0 - np.mean(class_means) if class_means else 1.0
    


class LogLikelihoodMicroLoss(LossStrategy):
    """Maximize Log-likelihood loss."""
    
    @staticmethod
    def prepare_targets(labels: np.ndarray, class_patterns: Optional[Dict], 
                       num_qubits: int) -> Dict:
        if class_patterns is None:
            raise ValueError("class_patterns must be set for log_likelihood_loss")
        return {
            label: Statevector.from_label(class_patterns[label])
            for label in labels
        }
    
    @staticmethod
    def compute(rho_list: List[np.ndarray], num_classes: int, 
                inner_products: List[np.ndarray], **kwargs) -> float:
        """Compute total log-likelihood across all data points."""
        num_samples = sum(len(fids) for fids in inner_products)

        total_loss = sum(
            np.sum(-np.log(np.clip(fids, FIDELITY_CLIP_MIN, FIDELITY_CLIP_MAX)))
            for fids in inner_products if len(fids) > 0
        ) / num_samples

        return total_loss
    

class LogLikelihoodMacroLoss(LossStrategy):
    """Maximize Log-likelihood loss."""
    
    @staticmethod
    def prepare_targets(labels: np.ndarray, class_patterns: Optional[Dict], 
                       num_qubits: int) -> Dict:
        if class_patterns is None:
            raise ValueError("class_patterns must be set for log_likelihood_loss")
        return {
            label: Statevector.from_label(class_patterns[label])
            for label in labels
        }
    
    @staticmethod
    def compute(rho_list: List[np.ndarray], num_classes: int, 
                inner_products: List[np.ndarray], **kwargs) -> float:
        """Compute total log-likelihood across all data points."""
        total_loss = [
            np.mean(-np.log(np.clip(fids, FIDELITY_CLIP_MIN, FIDELITY_CLIP_MAX)))
            for fids in inner_products if len(fids) > 0
        ]

        return np.mean(total_loss) if total_loss else 0.0



# Loss strategy registry
LOSS_STRATEGIES = {
    'hilbert_schmidt': HilbertSchmidtLoss,
    'pre_determine_inner': InnerLoss,
    'log_likelihood_macro': LogLikelihoodMacroLoss,
}


class TrainableQuantumFeatureMap:
    """Trainable Quantum Feature Map implementation."""

    def __init__(self, depth: int = 1, type_ansatz: str = "RealAmplitudes", 
                 type_loss: str = "trace_distance", warmup_iterations: int = 200):
        self.depth = depth
        self.ansatz_type = type_ansatz
        self.optimizer = None
        self.type_loss = type_loss
        self.warmup_iterations = warmup_iterations

        self.num_qubits = None
        self.init_theta = None
        self.num_classes = None
        self.unique_labels = None
        self.circuit = None
        self.class_patterns = None
        self.loss_strategy = None
        self.target_states = None

        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None

        self.optimal_params = None
        self.optimal_value = None
        self.best_params = None
        self.best_val_acc = 0.0
        
        # Best parameters for each metric
        self.best_params_qc_trace = None
        self.best_val_acc_qc_trace = 0.0
        self.best_rhos_qc_trace = None
        self.best_params_qc_hs = None
        self.best_val_acc_qc_hs = 0.0
        self.best_rhos_qc_hs = None
        self.best_params_qc_overlap = None
        self.best_val_acc_qc_overlap = 0.0
        self.best_rhos_qc_overlap = None

        # Initialize histories
        self.loss_history = []
        self.rhos = []
        self.self_overlaps = [] 
        self.pairwise_overlaps = [] 
        self.pairwise_distances = [] 
        self.train_accuracy_history = []
        self.accuracy_iteration_indices = []
        self.val_accuracy_history = []
        self.train_accuracy_qc_trace_history = []
        self.val_accuracy_qc_trace_history = []
        self.train_accuracy_qc_hs_history = []
        self.val_accuracy_qc_hs_history = []
        self.train_accuracy_qc_overlap_history = []
        self.val_accuracy_qc_overlap_history = []
        self.iteration_counter = 0


    def choose_class_patterns(self, X: np.ndarray, y: np.ndarray) -> Dict[int, str]:
        """
        Assign optimal basis states to each class based on inner product maximization.
        
        Args:
            X: Training data features
            y: Training data labels
            
        Returns:
            Dictionary mapping class labels to optimal basis state strings
        """
        unique_labels = np.unique(y)
        num_basis_states = 2**self.num_qubits
        
        if len(unique_labels) > num_basis_states:
            raise ValueError(
                f"Number of classes ({len(unique_labels)}) exceeds "
                f"available basis states ({num_basis_states})"
            )
        
        # Generate all basis states efficiently
        basis_states = [format(n, f'0{self.num_qubits}b') for n in range(num_basis_states)]
        
        # Cache class indices once to avoid repeated np.where() calls
        class_indices = {label: np.where(y == label)[0] for label in unique_labels}
        
        # Use all data points from each class
        class_samples = {
            label: X[idx]
            for label, idx in class_indices.items()
        }

        # Pre-compute circuit parameters structure
        data_params = self.circuit.parameters[:X.shape[1]]
        theta_params = self.circuit.parameters[X.shape[1]:]
        
        # Pre-assign theta parameters (constant across all samples) using dict(zip)
        circuit_with_theta = self.circuit.assign_parameters(
            dict(zip(theta_params, self.init_theta))
        )
        
        # Calculate average inner products between class samples and basis states
        inner_products = np.zeros((len(unique_labels), num_basis_states))
        for label_idx, label in enumerate(unique_labels):
            samples = class_samples[label]
            
            # Accumulate probabilities across all samples
            prob_sum = np.zeros(num_basis_states)
            for x in samples:
                # Build parameter dictionary using dict(zip) for efficiency
                bound_circ = circuit_with_theta.assign_parameters(dict(zip(data_params, x)))

                # Get statevector and compute probabilities for all basis states at once
                psi = Statevector.from_instruction(bound_circ)
                prob_sum += psi.probabilities()
            
            # Average probabilities across samples
            inner_products[label_idx, :] = prob_sum / len(samples)

        # Greedy assignment: iteratively select highest inner product
        final_assignment = {}
        
        for _ in range(len(unique_labels)):
            # Find maximum inner product
            row_idx, col_idx = np.unravel_index(np.argmax(inner_products), inner_products.shape)
            label = unique_labels[row_idx]
            
            # Assign basis state to class
            final_assignment[label] = basis_states[col_idx]
            
            # Mark this class and basis state as used
            inner_products[row_idx, :] = -np.inf
            inner_products[:, col_idx] = -np.inf

        return dict(sorted(final_assignment.items()))


    def _compute_class_density_matrices(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        circuit_with_theta: QuantumCircuit,
        data_params: List,
        target_states: Optional[Dict] = None
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Compute density matrices and optional inner products for each class.
        
        Args:
            X: Input data
            y: Labels
            circuit_with_theta: Quantum circuit with theta parameters bound
            data_params: Data parameter objects
            target_states: Optional target states for inner product computation
            
        Returns:
            Tuple of (density matrices list, fidelities array for all data points)
        """
        rho_list = []
        all_fidelities = []
        
        # Cache class indices to avoid repeated np.where() calls
        class_indices = {label: np.where(y == label)[0] for label in self.unique_labels}
        
        for label in self.unique_labels:
            idx = class_indices[label]
            samples = X[idx]
            num_samples_in_class = len(idx)
            
            # Compute statevectors for all samples in this class using dict(zip)
            statevectors = [
                Statevector.from_instruction(
                    circuit_with_theta.assign_parameters(dict(zip(data_params, sample)))
                )
                for sample in samples
            ]
            
            # Compute density matrix
            rho = sum(
                np.outer(psi.data, np.conj(psi.data)) for psi in statevectors
            ) / num_samples_in_class
            rho_list.append(rho)
            
            # Compute individual fidelities for each data point
            class_fids = []
            if target_states is not None:
                target_data = target_states[label]
                # Target data already extracted in prepare_targets for Statevectors
                if hasattr(target_data, 'data'):
                    target_data = target_data.data

                if target_data.ndim == 1:
                    class_fids = [
                        np.abs(np.vdot(psi.data, target_data))**2
                        for psi in statevectors
                    ]

            all_fidelities.append(np.array(class_fids))

        return rho_list, all_fidelities


    def _store_history(self, rho_list: List[np.ndarray], loss: float) -> None:
        """Store training history metrics."""
        # Self-overlaps (purity)
        self_overlaps = [
            float(np.real(np.trace(rho @ rho)))
            for rho in rho_list
        ]
        
        # Pairwise metrics
        pairwise_overlaps_dict = {
            f"{self.unique_labels[i]}_{self.unique_labels[j]}": float(np.real(
                np.trace(sqrtm(sqrtm(rho_list[i]) @ rho_list[j] @ sqrtm(rho_list[i])))
            ))
            for i in range(self.num_classes - 1)
            for j in range(i + 1, self.num_classes)
        }
        
        pairwise_distances_dict = {
            f"{self.unique_labels[i]}_{self.unique_labels[j]}": float(np.real(
                0.5 * np.sum(np.abs(np.linalg.eigh(rho_list[i] - rho_list[j])[0]))
            ))
            for i in range(self.num_classes - 1)
            for j in range(i + 1, self.num_classes)
        }
        
        self.rhos.append(rho_list)
        self.loss_history.append(loss)
        self.self_overlaps.append(self_overlaps)
        self.pairwise_overlaps.append(pairwise_overlaps_dict)
        self.pairwise_distances.append(pairwise_distances_dict)


    def _update_best_params(self, theta: np.ndarray, val_acc: float, 
                           metric_name: str, rho_list: List[np.ndarray]) -> None:
        """Update best parameters for a specific metric."""
        val_attr = f'best_val_acc_qc_{metric_name}'
        param_attr = f'best_params_qc_{metric_name}'
        rho_attr = f'best_rhos_qc_{metric_name}'
        
        current_best = getattr(self, val_attr)
        if val_acc > current_best:
            setattr(self, val_attr, val_acc)
            setattr(self, param_attr, theta.copy())
            setattr(self, rho_attr, [rho.copy() for rho in rho_list])


    def _track_accuracies(self, X: np.ndarray, y: np.ndarray, theta: np.ndarray,
                         circuit_template: QuantumCircuit, rho_list: List[np.ndarray]) -> None:
        """Track various accuracy metrics."""
        from src.kernel_estimate import KernelMatrix
        from src.classifier import QuantumClassifier, ClassicalClassifier
        
        # Kernel-based accuracy (hyperplane classifier)
        kernel_train = KernelMatrix.compute_kernel_matrix_with_inner_products(
            X, X, theta, circuit_template
        )
        kernel_val = KernelMatrix.compute_kernel_matrix_with_inner_products(
            self.X_val, X, theta, circuit_template
        )
        train_acc, val_acc = ClassicalClassifier.calculate_accuracy_fix(
            kernel_train, kernel_val, y, self.y_val
        )
        
        self.train_accuracy_history.append(train_acc)
        self.val_accuracy_history.append(val_acc)
        
        if self.iteration_counter >= self.warmup_iterations and val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_params = theta.copy()
        
        # Quantum classifier accuracies
        quantum_classifier = QuantumClassifier()
        quantum_classifier.fit(rho_list, circuit_template, theta)
        
        # Track each metric
        metrics_config = [
            ('trace_distance', 'trace', self.train_accuracy_qc_trace_history, 
             self.val_accuracy_qc_trace_history),
            ('hilbert_schmidt', 'hs', self.train_accuracy_qc_hs_history,
             self.val_accuracy_qc_hs_history),
            ('overlap', 'overlap', self.train_accuracy_qc_overlap_history,
             self.val_accuracy_qc_overlap_history),
        ]
        
        for metric_name, short_name, train_hist, val_hist in metrics_config:
            train_acc_metric = quantum_classifier.score(X, y, metric=metric_name)
            val_acc_metric = quantum_classifier.score(self.X_val, self.y_val, metric=metric_name)
            
            train_hist.append(train_acc_metric)
            val_hist.append(val_acc_metric)
            
            if self.iteration_counter >= self.warmup_iterations:
                self._update_best_params(theta, val_acc_metric, short_name, rho_list)


    def _loss(self, theta: np.ndarray, store_history: bool = True) -> float:
        """
        Compute the loss function using pre-configured strategy.
        
        Args:
            theta: Trainable parameters
            store_history: Whether to store history
            
        Returns:
            Loss value
        """
        # Prepare circuit with theta parameters
        theta_params = self.circuit.parameters[self.X_train.shape[1]:]
        theta_dict = {theta_params[k]: theta[k] for k in range(len(theta_params))}
        circuit_with_theta = self.circuit.assign_parameters(theta_dict)
        
        # Compute density matrices (using pre-computed target states if available)
        data_params = self.circuit.parameters[:self.X_train.shape[1]]
        rho_list, fidelities = self._compute_class_density_matrices(
            self.X_train, self.y_train, circuit_with_theta, data_params, self.target_states
        )
        
        # Compute loss using pre-configured strategy
        loss = self.loss_strategy.compute(
            rho_list, self.num_classes, 
            inner_products=fidelities,
            target_states=self.target_states,
            labels=self.unique_labels
        )
        loss = float(np.real(loss))
        
        # Store history and track accuracies
        if store_history:
            self._store_history(rho_list, loss)
            
            if self.iteration_counter % 100 == 0:
                self._track_accuracies(self.X_train, self.y_train, theta, self.circuit, rho_list)
                self.accuracy_iteration_indices.append(self.iteration_counter)
            
            self.iteration_counter += 1
        
        return loss


    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: np.ndarray, y_val: np.ndarray, optimizer,
            init_theta: Optional[np.ndarray] = None, 
            circuit: Optional[QuantumCircuit] = None) -> None:
        """
        Fit the quantum feature map to the data.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            optimizer: Optimization algorithm
            init_theta: Initial parameters (optional)
            circuit: Custom quantum circuit (optional)
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.optimizer = optimizer

        self.num_qubits = X_train.shape[1]
        self.num_classes = len(np.unique(y_train))
        self.unique_labels = np.unique(y_train)
        print(f"Number of qubits: {self.num_qubits}, Number of classes: {self.num_classes}")


        # Set up circuit
        if circuit is None:
            self._set_circuit(type=self.ansatz_type)
        else:
            self.circuit = circuit

        # Initialize parameters
        if init_theta is None:
            num_params = len(self.circuit.parameters) - self.num_qubits
            self.init_theta = np.random.uniform(-np.pi, np.pi, num_params)
        else:
            self.init_theta = init_theta

        print(f"Initial theta: {self.init_theta}")
        
        # Validate loss strategy
        if self.type_loss in LOSS_STRATEGIES:
            print(f"Using loss strategy: {LOSS_STRATEGIES[self.type_loss].__name__}")
        else:
            raise ValueError(f"Unknown loss type: {self.type_loss}")
        
        # Initialize loss strategy once
        self.loss_strategy = LOSS_STRATEGIES.get(self.type_loss)
        if self.loss_strategy is None:
            raise ValueError(f"Unknown loss type: {self.type_loss}")
        
        # Compute class patterns once if needed
        if self.type_loss in {"pre_determine_inner", "log_likelihood_micro", "log_likelihood_macro", "kl_divergence"}:
            self.class_patterns = self.choose_class_patterns(X_train, y_train)
            print(f"Class patterns: {self.class_patterns}")
        elif self.type_loss == "inner_product":
            if self.num_classes == 2:
                self.class_patterns = {self.unique_labels[0]: '00', self.unique_labels[1]: '01'}
            elif self.num_classes == 3:
                self.class_patterns = {self.unique_labels[0]: '00', self.unique_labels[1]: '01', 
                                      self.unique_labels[2]: '10'}
            elif self.num_classes == 4:
                self.class_patterns = {self.unique_labels[0]: '00', self.unique_labels[1]: '01',
                                      self.unique_labels[2]: '10', self.unique_labels[3]: '11'}
            else:
                raise ValueError(f"inner_product only supports 2-4 classes, got {self.num_classes}")
            print(f"Class patterns: {self.class_patterns}")
        else:
            self.class_patterns = None
        
        # Prepare target states
        self.target_states = self.loss_strategy.prepare_targets(
            self.unique_labels, self.class_patterns, self.num_qubits
        )
        
        # Clear previous history
        self._reset_history()

        # Perform optimization (only theta changes during training)
        result = self.optimizer.minimize(self._loss, x0=self.init_theta)

        # Store results
        self.optimal_params = result.x
        self.optimal_value = result.fun

        print(f"\nOptimization completed!")
        print(f"Optimal value: {self.optimal_value:.6f}")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")


    def _reset_history(self) -> None:
        """Reset all history tracking variables."""
        self.loss_history = []
        self.rhos = []
        self.self_overlaps = []
        self.pairwise_overlaps = []
        self.pairwise_distances = []
        self.train_accuracy_history = []
        self.accuracy_iteration_indices = []
        self.val_accuracy_history = []
        self.train_accuracy_qc_trace_history = []
        self.val_accuracy_qc_trace_history = []
        self.train_accuracy_qc_hs_history = []
        self.val_accuracy_qc_hs_history = []
        self.train_accuracy_qc_overlap_history = []
        self.val_accuracy_qc_overlap_history = []
        self.best_val_acc = 0.0
        self.best_params = None
        self.best_val_acc_qc_trace = 0.0
        self.best_params_qc_trace = None
        self.best_rhos_qc_trace = None
        self.best_val_acc_qc_hs = 0.0
        self.best_params_qc_hs = None
        self.best_rhos_qc_hs = None
        self.best_val_acc_qc_overlap = 0.0
        self.best_params_qc_overlap = None
        self.best_rhos_qc_overlap = None
        self.iteration_counter = 0


    def _set_circuit(self, type: str = 'EfficientSU2') -> None:
        """
        Set a custom quantum circuit.
        
        Args:
            type: Circuit type ('Universal_circuit', 'RealAmplitudes', 'EfficientSU2')
        """
        from src.paramertrized_circuit import ParametrizedCircuit


        if type == 'Universal':
            self.circuit = ParametrizedCircuit.Universal_circuit(self.num_qubits, self.depth)
        elif type == 'RealAmplitudes':
            self.circuit = ParametrizedCircuit.RealAmplitudes_circuit(self.num_qubits, self.depth)
        elif type == 'EfficientSU2':
            self.circuit = ParametrizedCircuit.EfficientSU2_circuit(self.num_qubits, self.depth)
        else:
            raise ValueError(f"Unknown circuit type: {type}")


    def get_optimal_params(self) -> np.ndarray:
        """Get the optimal parameters after fitting."""
        return self.optimal_params

    def get_optimal_value(self) -> float:
        """Get the optimal loss value after fitting."""
        return self.optimal_value
    
    def get_circuit(self) -> QuantumCircuit:
        """Get the quantum circuit."""
        return self.circuit
    
    def get_rhos(self) -> Optional[List[np.ndarray]]:
        """Get the list of density matrices (rhos) computed during training."""
        if not self.rhos:
            return None
        return self.rhos[-1]
    
    def get_best_rhos_qc_trace(self) -> Optional[List[np.ndarray]]:
        """Get the best rhos for trace_distance metric."""
        return self.best_rhos_qc_trace
    
    def get_best_rhos_qc_hs(self) -> Optional[List[np.ndarray]]:
        """Get the best rhos for hilbert_schmidt metric."""
        return self.best_rhos_qc_hs
    
    def get_best_rhos_qc_overlap(self) -> Optional[List[np.ndarray]]:
        """Get the best rhos for overlap metric."""
        return self.best_rhos_qc_overlap
    
    def set_optimizer(self, optimizer) -> None:
        """Set the optimizer."""
        self.optimizer = optimizer


if __name__ == "__main__":
    pass
