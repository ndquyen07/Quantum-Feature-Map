import numpy as np
from qiskit.quantum_info import Statevector

class QuantumClassifier:
    def __init__(self):
        self.rhoA = None
        self.rhoB = None
        self.circuit = None
        self.theta_params = None
    
    @staticmethod
    def create_pure_state(X, theta_params, circuit):

        n_samples = X.shape[0]
        n_features = X.shape[1]
        
        all_params = list(circuit.parameters)
        data_params = all_params[:n_features]
        theta_params_list = all_params[n_features:]
        

        theta_dict = {k: float(v) for k, v in zip(theta_params_list, theta_params)}
        
        rhos = []
        for i in range(n_samples):
            param_dict = theta_dict.copy()
            for k, param_obj in enumerate(data_params):
                param_dict[param_obj] = float(X[i, k])

            bound_circuit = circuit.assign_parameters(param_dict)
            sv = Statevector(bound_circuit).data
            
            rho = np.outer(sv, np.conj(sv))
            rhos.append(rho)
        
        return np.array(rhos)


    def fit(self, rhos, circuit, theta):
        self.rhos = rhos
        self.rhoA = rhos[0]
        self.rhoB = rhos[1]
        self.circuit = circuit
        self.theta_params = theta


    def predict(self, X_test, metric='trace_distance'):

        rho_test = self.create_pure_state(X_test, self.theta_params, self.circuit)
        
        predictions = []
        
        for i in range(len(rho_test)):
            rho_x = rho_test[i]
            
            if metric == 'trace_distance':
                score_A = -0.5 * np.sum(np.abs(np.linalg.eigh(rho_x - self.rhoA)[0]))
                score_B = -0.5 * np.sum(np.abs(np.linalg.eigh(rho_x - self.rhoB)[0]))
            elif metric == 'hilbert_schmidt':
                score_A = np.trace(rho_x @ self.rhoA).real - np.trace(self.rhoA @ self.rhoA).real
                score_B = np.trace(rho_x @ self.rhoB).real - np.trace(self.rhoB @ self.rhoB).real
            elif metric == 'overlap':
                score_A = np.trace(rho_x @ self.rhoA).real
                score_B = np.trace(rho_x @ self.rhoB).real

            
            if score_A > score_B:
                predictions.append(0)
            else:
                predictions.append(1)
                
        return np.array(predictions)
    
    
    def score(self, X_test, y_test, metric='trace_distance'):
        y_pred = self.predict(X_test, metric=metric)
        accuracy = np.mean(y_pred == y_test)
        return accuracy
    

class ClassicalClassifier:
    @staticmethod
    def evaluate_model(kernel_train, kernel_test, y_train, y_test):
        from sklearn.svm import SVC
        from sklearn.model_selection import GridSearchCV
        
        # Use GridSearchCV to select optimal C parameter
        c_values = [0.001, 0.01, 0.1, 1.0, 10.0, 20.0, 30.0, 40.0, 50.0,
                    60.0, 70.0, 80.0, 90.0, 100.0, 200.0, 300.0, 400.0,
                    500.0, 600.0, 700.0, 800.0, 900.0, 1000.0]
        param_grid = {'C': c_values}
        
        svc = SVC(kernel='precomputed')
        grid_search = GridSearchCV(svc, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(kernel_train, y_train)
        
        # Get the best model and parameters
        best_model = grid_search.best_estimator_
        best_c = grid_search.best_params_['C']
        best_score = grid_search.best_score_
        
        # Evaluate on test set with best model
        test_acc = best_model.score(kernel_test, y_test)
        
        return best_score, test_acc, best_model, best_c


    @staticmethod
    def calculate_accuracy_fix(kernel_train, kernel_val, y_train, y_val):
        from sklearn.svm import SVC
            
        svc = SVC(kernel='precomputed', C=1.0)
        svc.fit(kernel_train, y_train)

        train_acc = svc.score(kernel_train, y_train)
        val_acc = svc.score(kernel_val, y_val)
        
        return train_acc, val_acc
