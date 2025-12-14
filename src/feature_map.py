import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

class TrainableQuantumFeatureMap:
    """Trainable Quantum Feature Map (TQFM) implementation."""

    def __init__(self, depth: int = 1, type_ansatz: str = "RealAmplitudes", type_loss: str = "trace_distance", warmup_iterations: int = 200):
        self.depth = depth
        self.type = type_ansatz
        self.optimizer = None
        self.type_loss = type_loss
        self.warmup_iterations = warmup_iterations

        self.num_qubits = None
        self.init_theta = None
        self.num_classes = None
        self.circuit = None

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

        self.loss_history = []
        self.rhos = []
        self.ovlA = []
        self.ovlB = []
        self.cross_ovl = []
        self.distance = []
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


    def _loss(self, theta, X, y, circuit_template, num_classes, store_history: bool = True):
        """
        Compute the loss function
        """
        if self.type_loss == "inner_loss":
            if num_classes == 2:
                class_patterns = {0: '0', 1: '1'}
            elif num_classes == 3:
                class_patterns = {0: '00', 1: '01', 2: '10'}
            elif num_classes == 4:
                class_patterns = {0: '00', 1: '01', 2: '10', 3: '11'}
            elif num_classes == 5:
                class_patterns = {0: '000', 1: '001', 2: '010', 3: '011', 4: '100'}
            elif num_classes == 6:
                class_patterns = {0: '000', 1: '001', 2: '010', 3: '011', 4: '100', 5: '101'}
            elif num_classes == 7:
                class_patterns = {0: '000', 1: '001', 2: '010', 3: '011', 4: '100', 5: '101', 6: '110'}
            elif num_classes == 8:
                class_patterns = {0: '000', 1: '001', 2: '010', 3: '011', 4: '100', 5: '101', 6: '110', 7: '111'}
            elif num_classes == 9:
                class_patterns = {0: '0000', 1: '0001', 2: '0010', 3: '0011', 4: '0100', 5: '0101', 6: '0110', 7: '0111', 8: '1000'}
            elif num_classes == 10:
                class_patterns = {0: '0000', 1: '0001', 2: '0010', 3: '0011', 4: '0100', 5: '0101', 6: '0110', 7: '0111', 8: '1000', 9: '1001'}


        loss = 0.0
        rho_list = []
        for j in range(num_classes):
            # Select samples of class j
            idx = np.where(y == j)[0]
            M_j = len(idx)
            
            rho = 0.0
            class_loss = 0.0
            for i in idx:
                # Bind Data parameters
                param_dict = {}
                for k in range(X.shape[1]):
                    param_dict[circuit_template.parameters[k]] = X[i, k]
                # Bind Theta parameters
                theta_params = list(circuit_template.parameters)[X.shape[1]:]
                for k, t in enumerate(theta):
                    param_dict[theta_params[k]] = t
                # Get statevector
                psi = Statevector.from_instruction(circuit_template.assign_parameters(param_dict))
                # Get rho
                rho += 1/M_j * np.outer(psi.data, np.conj(psi.data))

                if self.type_loss == "inner_loss":
                    # Create basis state for class j
                    pattern = class_patterns[j]
                    if self.num_qubits > len(pattern):
                        # Pad with zeros for remaining qubits
                        remaining_qubits = self.num_qubits - len(pattern)
                        pattern = pattern + '0' * remaining_qubits
                    
                    y_j = Statevector.from_label(pattern)
                    inner_product = np.abs(np.vdot(psi.data, y_j.data))**2
                    class_loss += inner_product

            if self.type_loss == "inner_loss":        
                loss += class_loss / M_j

            rho_list.append(rho)
        
        self_ovlA = np.trace(rho_list[0] @ rho_list[0])
        self_ovlB = np.trace(rho_list[1] @ rho_list[1])
        cross_ovl = np.trace(rho_list[0] @ rho_list[1])
        distance = 0.5 * np.sum(np.abs(np.linalg.eigh(rho_list[0] - rho_list[1])[0]))

        
        # Compute loss based on selected type
        if self.type_loss == "trace_distance":
            loss = 1 - 0.5 * np.sum(np.abs(np.linalg.eigh(rho_list[0] - rho_list[1])[0]))
        elif self.type_loss == "hilbert_schmidt":
            loss = 1 - (0.5*(self_ovlA + self_ovlB) - cross_ovl)
        elif self.type_loss == "inner_loss":
            loss = 1 - (loss / num_classes)


        loss = float(np.real(loss))

        # Store loss history
        if store_history:
            self.rhos.append(rho_list)
            self.loss_history.append(loss)
            self.ovlA.append(float(np.real(self_ovlA)))
            self.ovlB.append(float(np.real(self_ovlB)))
            self.cross_ovl.append(float(np.real(cross_ovl)))
            self.distance.append(float(np.real(distance)))


            # Calculate training accuracy every 100 iterations
            if self.iteration_counter % 100 == 0:
                from src.kernel_estimate import KernelMatrix
                from src.classifier import QuantumClassifier, ClassicalClassifier
                
                # Kernel-based accuracy (hyperplane)
                kernel_train = KernelMatrix.compute_kernel_matrix_with_inner_products(X, X, theta, circuit_template)
                kernel_val = KernelMatrix.compute_kernel_matrix_with_inner_products(self.X_val, X, theta, circuit_template)
                
                train_acc, val_acc = ClassicalClassifier.calculate_accuracy_fix(kernel_train, kernel_val, y, self.y_val)

                self.train_accuracy_history.append(train_acc)
                self.val_accuracy_history.append(val_acc)

                if self.iteration_counter >= self.warmup_iterations:
                    if self.best_val_acc <= val_acc:
                        self.best_val_acc = val_acc
                        self.best_params = theta.copy()

                # Quantum Classifier accuracy with different metrics
                qc = QuantumClassifier()
                qc.fit(rho_list, circuit_template, theta)
                
                # Trace distance metric
                train_acc_qc_trace = qc.score(X, y, metric='trace_distance')
                val_acc_qc_trace = qc.score(self.X_val, self.y_val, metric='trace_distance')
                self.train_accuracy_qc_trace_history.append(train_acc_qc_trace)
                self.val_accuracy_qc_trace_history.append(val_acc_qc_trace)
                if self.iteration_counter >= self.warmup_iterations:
                    if self.best_val_acc_qc_trace <= val_acc_qc_trace:
                        self.best_val_acc_qc_trace = val_acc_qc_trace
                        self.best_params_qc_trace = theta.copy()
                        self.best_rhos_qc_trace = [rho.copy() for rho in rho_list]
                
                # Hilbert-Schmidt metric
                train_acc_qc_hs = qc.score(X, y, metric='hilbert_schmidt')
                val_acc_qc_hs = qc.score(self.X_val, self.y_val, metric='hilbert_schmidt')
                self.train_accuracy_qc_hs_history.append(train_acc_qc_hs)
                self.val_accuracy_qc_hs_history.append(val_acc_qc_hs)
                if self.iteration_counter >= self.warmup_iterations:
                    if self.best_val_acc_qc_hs <= val_acc_qc_hs:
                        self.best_val_acc_qc_hs = val_acc_qc_hs
                        self.best_params_qc_hs = theta.copy()
                        self.best_rhos_qc_hs = [rho.copy() for rho in rho_list]
                
                # Overlap metric
                train_acc_qc_overlap = qc.score(X, y, metric='overlap')
                val_acc_qc_overlap = qc.score(self.X_val, self.y_val, metric='overlap')
                self.train_accuracy_qc_overlap_history.append(train_acc_qc_overlap)
                self.val_accuracy_qc_overlap_history.append(val_acc_qc_overlap)
                if self.iteration_counter >= self.warmup_iterations:
                    if self.best_val_acc_qc_overlap <= val_acc_qc_overlap:
                        self.best_val_acc_qc_overlap = val_acc_qc_overlap
                        self.best_params_qc_overlap = theta.copy()
                        self.best_rhos_qc_overlap = [rho.copy() for rho in rho_list]

                self.accuracy_iteration_indices.append(self.iteration_counter)
            self.iteration_counter += 1

        return loss


    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, optimizer, init_theta: np.ndarray=None, circuit: QuantumCircuit=None) -> float:
        """Fit the quantum feature map to the data."""
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        
        self.optimizer = optimizer

        self.num_qubits = X_train.shape[1]
        self.num_classes = len(np.unique(y_train))
        print(f"Number of qubits: {self.num_qubits}, Number of classes: {self.num_classes}")

        if circuit is None:
            self._set_circuit(type=self.type)
        else:
            self.circuit = circuit

        if init_theta is None:
            num_params = len(self.circuit.parameters) - self.num_qubits
            self.init_theta = np.random.uniform(-np.pi, np.pi, num_params)
            # self.init_theta = np.zeros((num_params,))
        else:
            self.init_theta = init_theta

        print(self.init_theta)
        
        # Clear previous loss history
        self.loss_history = []
        self.rhos = []
        self.ovlA = []
        self.ovlB = []
        self.cross_ovl = []
        self.distance = []
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

        # Perform optimization using the optimizer parameter
        fun = lambda theta: self._loss(theta, self.X_train, self.y_train, self.circuit, self.num_classes, store_history=True)
        result = self.optimizer.minimize(fun, x0=self.init_theta)

        # Store results
        self.optimal_params = result.x
        self.optimal_value = result.fun

        return



    def _set_circuit(self, type= 'EfficientSU2'):
        from src.paramertrized_circuit import ParametrizedCircuit

        """Set a custom quantum circuit."""
        if type == 'TwoLocal':
            self.circuit = ParametrizedCircuit.TwoLocal_circuit(self.num_qubits, self.depth)
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
    
    def get_rhos(self) -> list:
        """Get the list of density matrices (rhos) computed during training."""
        return self.rhos[-1]
    
    def get_best_rhos_qc_trace(self) -> list:
        """Get the best rhos for trace_distance metric."""
        return self.best_rhos_qc_trace
    
    def get_best_rhos_qc_hs(self) -> list:
        """Get the best rhos for hilbert_schmidt metric."""
        return self.best_rhos_qc_hs
    
    def get_best_rhos_qc_overlap(self) -> list:
        """Get the best rhos for overlap metric."""
        return self.best_rhos_qc_overlap
    
    def set_optimizer(self, optimizer: str, maxiter: int = 100):
        """Set the optimizer and its parameters."""
        self.optimizer = optimizer
        self.maxiter = maxiter






if __name__ == "__main__":
    pass
    