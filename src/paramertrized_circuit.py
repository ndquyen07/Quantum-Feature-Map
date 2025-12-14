from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

class ParametrizedCircuit:
    """Class to create parameterized quantum circuits (ansatz)."""

    @staticmethod
    def TwoLocal_circuit(num_qubits, depth) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)

        data_params = ParameterVector('x', length=num_qubits)
        # Calculate total theta parameters needed: num_qubits for each layer (depth + 1)
        num_gates = 2
        total_theta_params = (num_qubits * 2 + num_qubits * depth * num_gates) * 2
        theta_params = ParameterVector('θ', length=total_theta_params)
        param_idx = 0

        # Initial first layer
        for i in range(num_qubits):
            qc.ry(theta_params[param_idx] * data_params[i] + theta_params[param_idx+1], i)
            param_idx += 2
        for i in range(num_qubits):
            qc.rz(theta_params[param_idx] * data_params[i] + theta_params[param_idx+1], i)
            param_idx += 2
        
        # Layers of parameterized rotations and entangling gates
        for _ in range(depth):
            for i in range(num_qubits - 1):
                qc.cz(i, i + 1)
            for i in range(num_qubits):
                qc.ry(theta_params[param_idx] * data_params[i] + theta_params[param_idx+1], i)
                param_idx += 2
            for i in range(num_qubits):
                qc.rz(theta_params[param_idx] * data_params[i] + theta_params[param_idx+1], i)
                param_idx += 2
        return qc


    @staticmethod
    def RealAmplitudes_circuit(num_qubits, depth) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)

        data_params = ParameterVector('x', length=num_qubits)
        # Calculate total theta parameters needed: num_qubits for each layer (depth + 1)
        num_gates = 1
        total_theta_params = (num_qubits + num_qubits * depth * num_gates)*2
        theta_params = ParameterVector('θ', length=total_theta_params)
        param_idx = 0

        # Initial first layer
        for i in range(num_qubits):
            qc.ry(theta_params[param_idx] * data_params[i] + theta_params[param_idx+1], i)
            param_idx += 2

        # Layers of parameterized rotations and entangling gates
        for _ in range(depth):
            for i in range(num_qubits - 1):
                qc.cx(i, i + 1)
            for i in range(num_qubits):
                qc.ry(theta_params[param_idx] * data_params[i] + theta_params[param_idx+1], i)
                param_idx += 2

        return qc
    

    @staticmethod
    def EfficientSU2_circuit(num_qubits, depth) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)

        data_params = ParameterVector('x', length=num_qubits)
        # Calculate total theta parameters needed: num_qubits for each layer (depth + 1)
        num_gates = 2
        total_theta_params = (num_qubits * 2 + num_qubits * depth * num_gates) * 2
        theta_params = ParameterVector('θ', length=total_theta_params)
        param_idx = 0

        # Initial first layer
        for i in range(num_qubits):
            qc.ry(theta_params[param_idx] * data_params[i] + theta_params[param_idx+1], i)
            param_idx += 2
        for i in range(num_qubits):
            qc.rz(theta_params[param_idx] * data_params[i] + theta_params[param_idx+1], i)
            param_idx += 2
        
        # Layers of parameterized rotations and entangling gates
        for _ in range(depth):
            for i in range(num_qubits - 1):
                qc.cx(i, i + 1)
            for i in range(num_qubits):
                qc.ry(theta_params[param_idx] * data_params[i] + theta_params[param_idx+1], i)
                param_idx += 2
            for i in range(num_qubits):
                qc.rz(theta_params[param_idx] * data_params[i] + theta_params[param_idx+1], i)
                param_idx += 2
        return qc
    