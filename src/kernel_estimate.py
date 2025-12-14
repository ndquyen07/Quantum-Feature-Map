import numpy as np
from qiskit.quantum_info import Statevector


class KernelMatrix:
    """Class to compute the kernel matrix with optimized batch processing."""

    @staticmethod
    def _get_statevectors(X, theta_params, circuit):
        n_samples = X.shape[0]
        n_features = X.shape[1]
        
        # Get parameter lists - cache these
        data_params = list(circuit.parameters)[:n_features]
        theta_params_list = list(circuit.parameters)[n_features:]
        
        # Build theta dictionary once
        theta_dict = {}
        for param, val in zip(theta_params_list, theta_params):
            # Convert numpy types to Python float
            if hasattr(val, 'item'):  # numpy scalar
                theta_dict[param] = float(val.item())
            elif isinstance(val, (int, float)):
                theta_dict[param] = float(val)
            else:
                theta_dict[param] = float(val)
        
        
        statevectors = []
        for i in range(n_samples):
            param_dict = theta_dict.copy()
            for k, param in enumerate(data_params):
                param_dict[param] = X[i, k]
            
            sv = Statevector.from_instruction(circuit.assign_parameters(param_dict))
            statevectors.append(sv.data)
        
        return np.array(statevectors)

    @staticmethod
    def compute_kernel_matrix_with_inner_products(X1, X2, theta_params, circuit):
        """
        Compute the kernel matrix K_ij = |<psi(x_i, theta)|psi(x_j, theta)>|^2 using inner products.
        Returns:
            Kernel matrix of shape (n_samples_1, n_samples_2)
        """
        # Check if we can exploit symmetry (K[i,j] = K[j,i] for same dataset)
        is_symmetric = (X1 is X2 or np.array_equal(X1, X2))
        
        # Compute statevectors for X1
        statevectors_1 = KernelMatrix._get_statevectors(X1, theta_params, circuit)
        
        # For symmetric case, reuse statevectors
        if is_symmetric:
            statevectors_2 = statevectors_1
        else:
            statevectors_2 = KernelMatrix._get_statevectors(X2, theta_params, circuit)
        
        # Vectorized computation of kernel matrix
        inner_products = statevectors_1 @ statevectors_2.conj().T
        kernel_matrix = np.real(inner_products * inner_products.conj())
        
        # For symmetric matrices
        if is_symmetric:
            kernel_matrix = (kernel_matrix + kernel_matrix.T) / 2.0

            np.fill_diagonal(kernel_matrix, 1.0)
        
        return kernel_matrix


    @staticmethod
    def compute_kernel_matrix_with_inner_products_legacy(X1, X2, theta_params, circuit):
        """
        Original (slow) implementation kept for comparison purposes.
        Compute the kernel matrix K_ij = |<psi(x_i, theta)|psi(x_j, theta)>|^2.
        """
        kernel_matrix = np.zeros((X1.shape[0], X2.shape[0]))
        for i in range(X1.shape[0]):
            for j in range(X2.shape[0]):
                # Prepare statevector for x_i
                param_dict_i = {}
                for k in range(X1.shape[1]):
                    param_dict_i[circuit.parameters[k]] = X1[i, k]
                theta_params_list = list(circuit.parameters)[X1.shape[1]:]
                for k, t in enumerate(theta_params):
                    param_dict_i[theta_params_list[k]] = t
                sv_i = Statevector.from_instruction(circuit.assign_parameters(param_dict_i))

                # Prepare statevector for x_j
                param_dict_j = {}
                for k in range(X2.shape[1]):
                    param_dict_j[circuit.parameters[k]] = X2[j, k]
                for k, t in enumerate(theta_params):
                    param_dict_j[theta_params_list[k]] = t
                sv_j = Statevector.from_instruction(circuit.assign_parameters(param_dict_j))

                # Compute squared inner product
                kernel_matrix[i, j] = np.abs(np.vdot(sv_i.data, sv_j.data))**2

        return kernel_matrix


    
if __name__ == "__main__":
    pass
