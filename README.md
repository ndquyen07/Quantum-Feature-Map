# Trainable Quantum Feature Map (TQFM) for Quantum SVM

A quantum machine learning framework for classification tasks using trainable quantum feature maps with various distance metrics.

## 📋 Table of Contents
- [Overview](#overview)
- [Training Pipeline](#training-pipeline)
- [Project Structure](#project-structure)
- [Loss Functions](#loss-functions)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation Workflow](#evaluation-workflow)
- [Results](#results)

---

## 🎯 Overview

This project implements **Trainable Quantum Feature Maps (TQFM)** for binary and multi-class classification using quantum circuits. The key innovation is learning optimal quantum feature map parameters through different distance-based loss functions.

### Key Features
- **Three Loss Functions**: Trace Distance, Hilbert-Schmidt, Inner Loss
- **Flexible Ansätze**: RealAmplitudes, EfficientSU2, TwoLocal
- **Validation-Based Selection**: Best parameters chosen via validation set
- **Hybrid Classical-Quantum**: SVM with quantum kernels
- **Comprehensive Evaluation**: 5 different test sets for robust performance metrics

---

## 🔄 Training Pipeline

The training workflow follows a rigorous 3-step process to ensure generalization:

### **Step 1: Feature Map Training (Quantum Optimization)**
```
Train Set (100 samples) → Train quantum feature map parameters
Val Set (50 samples)   → Select best parameters based on validation accuracy
```

- **Purpose**: Learn optimal quantum circuit parameters `θ*` that maximize class separability
- **Process**: 
  - Minimize loss function using training data
  - Evaluate on validation set every N iterations
  - Select parameters `θ*` that achieve **highest validation accuracy** (not training accuracy)
- **Why Validation?**: Prevents overfitting to training data, ensures generalization

**Loss Functions**:
- **Trace Distance**: `L = 1 - 0.5 * Σ|λᵢ(ρ₀ - ρ₁)|`
- **Hilbert-Schmidt**: `L = 1 - [0.5(Tr(ρ₀²) + Tr(ρ₁²)) - Tr(ρ₀ρ₁)]`
- **Inner Loss**: `L = 1 - (1/K) Σⱼ (1/Mⱼ) Σᵢ |⟨ψᵢ|yⱼ⟩|²`

### **Step 2: SVM Hyperparameter Tuning (Classical Optimization)**
```
Train ∪ Val (150 samples) → Compute quantum kernel → GridSearchCV for optimal C
```

- **Purpose**: Find optimal SVM regularization parameter `C*`
- **Process**:
  1. Combine train and validation sets (150 total samples)
  2. Compute quantum kernel matrix using `θ*` from Step 1
  3. Use **5-Fold Cross-Validation (GridSearchCV)** to select best `C*`
  4. Search over: `C ∈ [0.001, 0.01, 0.1, 1, 10, ..., 1000]`
- **Why GridSearchCV?**: Robust hyperparameter selection through cross-validation

**Quantum Kernel Computation**:
```python
K(x, x') = |⟨ψ(x)|ψ(x')⟩|²
```
where `ψ(x) = U(x; θ*)|0⟩` is the quantum state encoded with optimal parameters.

### **Step 3: Model Evaluation (Test Phase)**
```
5 Test Sets × 200 samples each → Compute mean & std accuracy
```

- **Purpose**: Comprehensive generalization performance assessment
- **Process**:
  1. Use best parameters `θ*` and `C*` from Steps 1-2
  2. Evaluate on 5 independent test sets (different random seeds)
  3. Each test set: 200 samples with noise=0.2
  4. Report: **Mean ± Std accuracy** across all 5 test sets
- **Why 5 Test Sets?**: Reduces variance in performance estimates, more reliable results

---

## 📁 Project Structure

```
SVQSVM1/
├── src/
│   ├── feature_map.py          # TrainableQuantumFeatureMap class
│   ├── classifier.py           # QuantumClassifier & ClassicalClassifier
│   ├── kernel_estimate.py      # Quantum kernel matrix computation
│   ├── paramertrized_circuit.py # Circuit templates (RealAmplitudes, etc.)
│   ├── optimizer.py            # Optimizer configurations (COBYLA, SPSA, etc.)
│   └── utils.py                # Utility functions
├── experiment/
│   ├── analysis_losses.ipynb       # Depth-1 loss comparison
│   ├── analysis_losses_depth8.ipynb # Depth-8 analysis
│   └── analyze_moon_depth.ipynb    # Multi-depth analysis (1-8)
├── main_three_losses_moons.py      # Main training script
├── main_three_losses_parallel_moon.py  # Parallel training
├── results_tqfm_losses/            # Saved model results (.pkl files)
├── data/                           # Dataset storage
├── environment.yml                 # Conda environment
└── requirements.txt                # Python dependencies
```

---

## 📊 Loss Functions

### 1. **Trace Distance**
- **Formula**: Measures maximum distinguishability between quantum states
- **Range**: [0, 1], where 1 = perfectly distinguishable
- **Best for**: Maximum class separation

### 2. **Hilbert-Schmidt Distance**
- **Formula**: Frobenius norm between density matrices
- **Range**: [0, √2], normalized to [0, 1]
- **Best for**: Smooth optimization landscape

### 3. **Inner Loss**
- **Formula**: Average overlap with target computational basis states
- **Range**: [0, 1], where 1 = perfect encoding
- **Best for**: Direct encoding to computational basis

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Conda (recommended)

### Setup
```bash
# Clone repository
git clone https://github.com/ndquyen07/Quantum-Feature-Map.git
cd SVQSVM1

# Create conda environment
conda env create -f environment.yml
conda activate svqsvm

# Or use pip
pip install -r requirements.txt
```

### Dependencies
- `qiskit >= 1.0.0`
- `qiskit-algorithms`
- `scikit-learn`
- `numpy`
- `matplotlib`
- `seaborn`
- `pandas`

---

## 🚀 Usage

### Basic Training Example
```python
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from src.feature_map import TrainableQuantumFeatureMap
from src.paramertrized_circuit import ParametrizedCircuit
from qiskit_algorithms.optimizers import COBYLA

# 1. Prepare data
X, y = make_moons(n_samples=150, noise=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=50, random_state=42)

# 2. Initialize TQFM
tqfm = TrainableQuantumFeatureMap(
    depth=1,
    type_ansatz="RealAmplitudes",
    type_loss="hilbert_schmidt",
    warmup_iterations=200
)

# 3. Train (Step 1: Learn θ*)
circuit = ParametrizedCircuit.RealAmplitudes_circuit(X_train.shape[1], depth=1)
optimizer = COBYLA(maxiter=5000)
tqfm.fit(X_train, y_train, X_val, y_val, optimizer, circuit=circuit)

# Best parameters selected by validation accuracy
best_params = tqfm.best_params  # θ* with highest val_acc

# 4. Compute quantum kernel for SVM (Step 2: Find C*)
from src.kernel_estimate import KernelMatrix
X_combined = np.vstack([X_train, X_val])
y_combined = np.hstack([y_train, y_val])

kernel_train = KernelMatrix.compute_kernel_matrix_with_inner_products(
    X_combined, X_combined, best_params, circuit
)

# 5. Train SVM with GridSearchCV
from src.classifier import ClassicalClassifier
train_acc, _, model, best_C = ClassicalClassifier.evaluate_model(
    kernel_train, kernel_train, y_combined, y_combined
)

# 6. Evaluate on test sets (Step 3)
test_accuracies = []
for seed in range(1, 6):
    X_test, y_test = make_moons(n_samples=200, noise=0.2, random_state=seed)
    kernel_test = KernelMatrix.compute_kernel_matrix_with_inner_products(
        X_test, X_combined, best_params, circuit
    )
    _, test_acc, _, _ = ClassicalClassifier.evaluate_model(
        kernel_train, kernel_test, y_combined, y_test
    )
    test_accuracies.append(test_acc)

print(f"Mean Test Accuracy: {np.mean(test_accuracies)*100:.2f}% ± {np.std(test_accuracies)*100:.2f}%")
```

### Run Complete Experiment
```bash
# Train all three loss functions
python main_three_losses_moons.py

# Parallel training (faster)
python main_three_losses_parallel_moon.py
```

---

## 📈 Evaluation Workflow

### Data Split Strategy
```
Total Dataset: 350 samples
├─ Training Phase: 150 samples
│  ├─ Train: 100 samples (optimize θ)
│  └─ Val:    50 samples (select best θ*)
├─ SVM Tuning: 150 samples (train + val combined)
└─ Testing: 5 × 200 samples (evaluate generalization)
```

### Metrics Tracked During Training
- **Loss History**: Loss value at each optimization iteration
- **Training Accuracy**: SVM accuracy on training set (with quantum kernel)
- **Validation Accuracy**: SVM accuracy on validation set
- **Distance Metrics**: Trace distance between class density matrices
- **Overlap Metrics**: Self-overlap and cross-overlap values

### Analysis Notebooks
1. **`analysis_losses.ipynb`**: Compare three losses at depth=1
2. **`analyze_moon_depth.ipynb`**: Analyze loss behavior across depths 1-8
   - Mean loss vs depth (with error bars)
   - Mean training accuracy vs depth
   - Statistical comparison of stability

---

## 📊 Results

### Expected Output Structure
Each training run saves a `.pkl` file containing:
```python
{
    'best_params': θ*,              # Best parameters (from validation)
    'best_val_acc': float,          # Best validation accuracy
    'optimal_value': float,         # Final loss value
    'loss_history': List[float],    # Loss trajectory
    'train_accuracy_history': [...],# Training accuracy over time
    'val_accuracy_history': [...],  # Validation accuracy over time
    'circuit': QuantumCircuit,      # Circuit structure
    'iteration_counter': int,       # Total iterations
    # ... additional metrics
}
```

### Interpreting Results
- **Low Training Loss + High Test Accuracy**: Good generalization
- **Low Training Loss + Low Test Accuracy**: Overfitting (validation helps prevent this)
- **High Loss Variance Across Runs**: Unstable optimization (increase warmup_iterations)
- **Similar Performance Across Losses**: Problem may not benefit from quantum advantage

---

## 🔬 Key Design Decisions

### Why Use Validation Set?
- **Problem**: Training accuracy can be misleading (model may memorize training data)
- **Solution**: Select parameters based on **validation accuracy**
- **Result**: Better generalization to unseen test data

### Why Combine Train+Val for SVM?
- **Problem**: More training data → better SVM performance
- **Solution**: After selecting `θ*`, use all 150 samples for final SVM training
- **Result**: Maximum data utilization without compromising validation integrity

### Why 5 Test Sets?
- **Problem**: Single test set may not reflect true performance
- **Solution**: Average over 5 independent test sets
- **Result**: More robust and reliable performance estimates (mean ± std)

---

## 🧪 Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `depth` | 1 | Circuit depth (layers of parametrized gates) |
| `type_ansatz` | "RealAmplitudes" | Ansatz type (RealAmplitudes, EfficientSU2, TwoLocal) |
| `type_loss` | "trace_distance" | Loss function (trace_distance, hilbert_schmidt, inner_loss) |
| `warmup_iterations` | 200 | Iterations before tracking best validation params |
| `maxiter` | 5000 | Maximum optimizer iterations |
| `optimizer` | COBYLA | Optimization algorithm |

---

## 📖 References

1. Havlíček, V., et al. (2019). "Supervised learning with quantum-enhanced feature spaces." *Nature* 567, 209–212.
2. Schuld, M., & Killoran, N. (2019). "Quantum Machine Learning in Feature Hilbert Spaces." *Physical Review Letters* 122, 040504.

---

## 👨‍💻 Author

**Nguyen Dinh Quyen**  
GitHub: [@ndquyen07](https://github.com/ndquyen07)

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

Built with Qiskit and inspired by quantum machine learning research in quantum kernel methods and trainable quantum circuits.
