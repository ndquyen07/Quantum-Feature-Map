from src.feature_map import TrainableQuantumFeatureMap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def draw_circuit(tqfm : TrainableQuantumFeatureMap):
    """Draw the quantum circuit."""
    circuit_str = tqfm.circuit.draw(output='text', fold=-1)
    print(circuit_str)


def save_circuit(tqfm : TrainableQuantumFeatureMap, filename: str):
    """Save the quantum circuit to a file."""
    tqfm.circuit.draw('mpl', filename=filename)


def plot_loss(tqfm : TrainableQuantumFeatureMap):
    """Plot the loss history."""

    plt.plot(tqfm.loss_history)
    plt.xlabel("iteration times")
    plt.ylabel(r"values of $E(\theta)$")
    plt.plot(tqfm.loss_history, color='red', label=r"$E(\theta)$")
    plt.axhline(0, color='black', linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.show()


def save_plot_loss(tqfm : TrainableQuantumFeatureMap, filename: str):
    """Save the loss history to a figure."""

    plt.plot(tqfm.loss_history)
    plt.xlabel("iteration times")
    plt.ylabel(r"values of $E(\theta)$")
    plt.plot(tqfm.loss_history, color='red', label=r"$E(\theta)$")
    plt.axhline(0, color='black', linestyle='--', label=r"$L=0$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()




def plot_all_metrics_accuracy(tqfm: TrainableQuantumFeatureMap):
    """Plot training and validation accuracy history for all metrics (classical SVC and quantum classifiers)."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Classical SVC (kernel)
    ax1 = axes[0, 0]
    ax1.plot(tqfm.accuracy_iteration_indices, tqfm.train_accuracy_history, 
             marker='o', linestyle='-', linewidth=2, markersize=6,
             color='green', label='Training', alpha=0.8)
    ax1.plot(tqfm.accuracy_iteration_indices, tqfm.val_accuracy_history, 
             marker='s', linestyle='--', linewidth=2, markersize=6,
             color='blue', label='Validation', alpha=0.8)
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Classical SVC (kernel)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=10, loc='best')
    
    # QC trace_distance
    ax2 = axes[0, 1]
    if hasattr(tqfm, 'train_accuracy_qc_trace_history') and len(tqfm.train_accuracy_qc_trace_history) > 0:
        ax2.plot(tqfm.accuracy_iteration_indices, tqfm.train_accuracy_qc_trace_history, 
                 marker='o', linestyle='-', linewidth=2, markersize=6,
                 color='green', label='Training', alpha=0.8)
        ax2.plot(tqfm.accuracy_iteration_indices, tqfm.val_accuracy_qc_trace_history, 
                 marker='s', linestyle='--', linewidth=2, markersize=6,
                 color='red', label='Validation', alpha=0.8)
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('QC trace_distance', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=10, loc='best')
    
    # QC hilbert_schmidt
    ax3 = axes[1, 0]
    if hasattr(tqfm, 'train_accuracy_qc_hs_history') and len(tqfm.train_accuracy_qc_hs_history) > 0:
        ax3.plot(tqfm.accuracy_iteration_indices, tqfm.train_accuracy_qc_hs_history, 
                 marker='o', linestyle='-', linewidth=2, markersize=6,
                 color='green', label='Training', alpha=0.8)
        ax3.plot(tqfm.accuracy_iteration_indices, tqfm.val_accuracy_qc_hs_history, 
                 marker='s', linestyle='--', linewidth=2, markersize=6,
                 color='orange', label='Validation', alpha=0.8)
    ax3.set_xlabel('Iteration', fontsize=12)
    ax3.set_ylabel('Accuracy', fontsize=12)
    ax3.set_title('QC hilbert_schmidt', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend(fontsize=10, loc='best')
    
    # QC overlap
    ax4 = axes[1, 1]
    if hasattr(tqfm, 'train_accuracy_qc_overlap_history') and len(tqfm.train_accuracy_qc_overlap_history) > 0:
        ax4.plot(tqfm.accuracy_iteration_indices, tqfm.train_accuracy_qc_overlap_history, 
                 marker='o', linestyle='-', linewidth=2, markersize=6,
                 color='green', label='Training', alpha=0.8)
        ax4.plot(tqfm.accuracy_iteration_indices, tqfm.val_accuracy_qc_overlap_history, 
                 marker='s', linestyle='--', linewidth=2, markersize=6,
                 color='purple', label='Validation', alpha=0.8)
    ax4.set_xlabel('Iteration', fontsize=12)
    ax4.set_ylabel('Accuracy', fontsize=12)
    ax4.set_title('QC overlap', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.legend(fontsize=10, loc='best')
    
    plt.tight_layout()
    plt.show()


def save_all_metrics_accuracy(tqfm: TrainableQuantumFeatureMap, filename: str):
    """Save training and validation accuracy history for all metrics to a file."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Classical SVC (kernel)
    ax1 = axes[0, 0]
    ax1.plot(tqfm.accuracy_iteration_indices, tqfm.train_accuracy_history, 
             marker='o', linestyle='-', linewidth=2, markersize=6,
             color='green', label='Training', alpha=0.8)
    ax1.plot(tqfm.accuracy_iteration_indices, tqfm.val_accuracy_history, 
             marker='s', linestyle='--', linewidth=2, markersize=6,
             color='blue', label='Validation', alpha=0.8)
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Classical SVC (kernel)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=10, loc='best')
    
    # QC trace_distance
    ax2 = axes[0, 1]
    if hasattr(tqfm, 'train_accuracy_qc_trace_history') and len(tqfm.train_accuracy_qc_trace_history) > 0:
        ax2.plot(tqfm.accuracy_iteration_indices, tqfm.train_accuracy_qc_trace_history, 
                 marker='o', linestyle='-', linewidth=2, markersize=6,
                 color='green', label='Training', alpha=0.8)
        ax2.plot(tqfm.accuracy_iteration_indices, tqfm.val_accuracy_qc_trace_history, 
                 marker='s', linestyle='--', linewidth=2, markersize=6,
                 color='red', label='Validation', alpha=0.8)
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('QC trace_distance', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=10, loc='best')
    
    # QC hilbert_schmidt
    ax3 = axes[1, 0]
    if hasattr(tqfm, 'train_accuracy_qc_hs_history') and len(tqfm.train_accuracy_qc_hs_history) > 0:
        ax3.plot(tqfm.accuracy_iteration_indices, tqfm.train_accuracy_qc_hs_history, 
                 marker='o', linestyle='-', linewidth=2, markersize=6,
                 color='green', label='Training', alpha=0.8)
        ax3.plot(tqfm.accuracy_iteration_indices, tqfm.val_accuracy_qc_hs_history, 
                 marker='s', linestyle='--', linewidth=2, markersize=6,
                 color='orange', label='Validation', alpha=0.8)
    ax3.set_xlabel('Iteration', fontsize=12)
    ax3.set_ylabel('Accuracy', fontsize=12)
    ax3.set_title('QC hilbert_schmidt', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend(fontsize=10, loc='best')
    
    # QC overlap
    ax4 = axes[1, 1]
    if hasattr(tqfm, 'train_accuracy_qc_overlap_history') and len(tqfm.train_accuracy_qc_overlap_history) > 0:
        ax4.plot(tqfm.accuracy_iteration_indices, tqfm.train_accuracy_qc_overlap_history, 
                 marker='o', linestyle='-', linewidth=2, markersize=6,
                 color='green', label='Training', alpha=0.8)
        ax4.plot(tqfm.accuracy_iteration_indices, tqfm.val_accuracy_qc_overlap_history, 
                 marker='s', linestyle='--', linewidth=2, markersize=6,
                 color='purple', label='Validation', alpha=0.8)
    ax4.set_xlabel('Iteration', fontsize=12)
    ax4.set_ylabel('Accuracy', fontsize=12)
    ax4.set_title('QC overlap', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.legend(fontsize=10, loc='best')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"All metrics accuracy plot saved to {filename}")


def plot_metrics(tqfm : TrainableQuantumFeatureMap):
    """Plot overlaps and distance metrics for multi-class scenarios."""
    num_classes = len(tqfm.self_overlaps[0]) if tqfm.self_overlaps else 0
    
    if num_classes == 0:
        print("No overlap data available.")
        return
    
    # Get unique labels from the first pairwise_overlaps entry
    if tqfm.pairwise_overlaps:
        labels = sorted(set(int(k.split('_')[0]) for k in tqfm.pairwise_overlaps[0].keys()) | 
                       set(int(k.split('_')[1]) for k in tqfm.pairwise_overlaps[0].keys()))
    else:
        labels = list(range(num_classes))
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot self-overlaps
    ax1 = axes[0]
    for i in range(num_classes):
        self_ovl = [ovls[i] for ovls in tqfm.self_overlaps]
        ax1.plot(self_ovl, label=f'Class {labels[i]}', linewidth=2)
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Self-overlap', fontsize=12)
    ax1.set_title('Self-overlaps', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot pairwise cross-overlaps
    ax2 = axes[1]
    for key in tqfm.pairwise_overlaps[0].keys():
        cross_ovl = [ovls[key] for ovls in tqfm.pairwise_overlaps]
        ax2.plot(cross_ovl, label=f'Classes {key}', linewidth=2)
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Cross-overlap', fontsize=12)
    ax2.set_title('Pairwise Cross-overlaps', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot pairwise distances
    ax3 = axes[2]
    for key in tqfm.pairwise_distances[0].keys():
        dist = [dists[key] for dists in tqfm.pairwise_distances]
        ax3.plot(dist, label=f'Classes {key}', linewidth=2)
    ax3.set_xlabel('Iteration', fontsize=12)
    ax3.set_ylabel('Trace Distance', fontsize=12)
    ax3.set_title('Pairwise Trace Distances', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    plt.show()


def save_plot_metrics(tqfm: TrainableQuantumFeatureMap, filename: str):
    """Save overlaps and distance metrics plot for multi-class scenarios."""
    num_classes = len(tqfm.self_overlaps[0]) if tqfm.self_overlaps else 0
    
    if num_classes == 0:
        print("No overlap data available.")
        return
    
    # Get unique labels from the first pairwise_overlaps entry
    if tqfm.pairwise_overlaps:
        labels = sorted(set(int(k.split('_')[0]) for k in tqfm.pairwise_overlaps[0].keys()) | 
                       set(int(k.split('_')[1]) for k in tqfm.pairwise_overlaps[0].keys()))
    else:
        labels = list(range(num_classes))
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot self-overlaps
    ax1 = axes[0]
    for i in range(num_classes):
        self_ovl = [ovls[i] for ovls in tqfm.self_overlaps]
        ax1.plot(self_ovl, label=f'Class {labels[i]}', linewidth=2)
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Self-overlap', fontsize=12)
    ax1.set_title('Self-overlaps', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot pairwise cross-overlaps
    ax2 = axes[1]
    for key in tqfm.pairwise_overlaps[0].keys():
        cross_ovl = [ovls[key] for ovls in tqfm.pairwise_overlaps]
        ax2.plot(cross_ovl, label=f'Classes {key}', linewidth=2)
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Cross-overlap', fontsize=12)
    ax2.set_title('Pairwise Cross-overlaps', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot pairwise distances
    ax3 = axes[2]
    for key in tqfm.pairwise_distances[0].keys():
        dist = [dists[key] for dists in tqfm.pairwise_distances]
        ax3.plot(dist, label=f'Classes {key}', linewidth=2)
    ax3.set_xlabel('Iteration', fontsize=12)
    ax3.set_ylabel('Trace Distance', fontsize=12)
    ax3.set_title('Pairwise Trace Distances', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Metrics plot saved to {filename}")


def save_plot_train_accuracy(tqfm : TrainableQuantumFeatureMap, filename: str):
    """Save the training accuracy history to a figure."""

    plt.figure(figsize=(10, 6))
    plt.plot(tqfm.accuracy_iteration_indices, tqfm.train_accuracy_history, 
            marker='o', linestyle='-', color='green', label='Training Accuracy')
    plt.xlabel('Iteration')
    plt.ylabel('Training Accuracy')
    plt.title('Training Accuracy vs Iteration')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_landscape_1d(tqfm : TrainableQuantumFeatureMap, param_idx: int = 0, grid_points: int = 100, 
                            param_range: float = np.pi, theta_base: np.ndarray = None):
    """
    Plot the loss landscape by varying a single parameter (1D plot).
    
    Args:
        param_idx: Index of the parameter to vary
        grid_points: Number of grid points
        param_range: Range to vary parameter around the base value
        theta_base: Base parameter values. If None, uses optimal_params or init_theta
    """
    if theta_base is None:
        if tqfm.optimal_params is not None:
            theta_base = tqfm.optimal_params.copy()
        elif tqfm.init_theta is not None:
            theta_base = tqfm.init_theta.copy()
        else:
            raise ValueError("No parameter values available. Please fit the model first or provide theta_base.")
    
    # Create parameter range
    param_center = theta_base[param_idx]
    param_vals = np.linspace(param_center - param_range, param_center + param_range, grid_points)
    loss_vals = np.zeros(grid_points)
    
    # Compute loss for each parameter value
    print(f"Computing 1D loss landscape ({grid_points} points)...")
    for i in range(grid_points):
        theta_temp = theta_base.copy()
        theta_temp[param_idx] = param_vals[i]
        loss_vals[i] = tqfm._loss(theta_temp, tqfm.X_train, tqfm.y_train, 
                                    tqfm.circuit, tqfm.num_classes, store_history=False)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(param_vals, loss_vals, linewidth=2, color='blue', label='Loss')
    ax.set_xlabel(f'θ[{param_idx}]', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'Loss Landscape for Parameter θ[{param_idx}]', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Mark initial point if available
    if tqfm.init_theta is not None:
        init_loss = tqfm._loss(theta_base, tqfm.X_train, tqfm.y_train, tqfm.circuit, tqfm.num_classes, store_history=False)
        ax.scatter([tqfm.init_theta[param_idx]], [init_loss],
                    color='yellow', s=150, marker='o', 
                    edgecolors='black', linewidths=2,
                    label='Initial', zorder=5)
    
    # Mark optimal point if available
    if tqfm.optimal_params is not None:
        ax.scatter([tqfm.optimal_params[param_idx]], [tqfm.optimal_value], 
                    color='red', s=200, marker='*', 
                    edgecolors='black', linewidths=2,
                    label=f'Optimal (Loss={tqfm.optimal_value:.4f})', zorder=5)
    
    ax.legend(fontsize=10, loc='best')
    plt.tight_layout()
    plt.show()


def save_landscape_1d(tqfm : TrainableQuantumFeatureMap, filename: str, param_idx: int = 0, 
                            grid_points: int = 100, param_range: float = np.pi, 
                            theta_base: np.ndarray = None):
    """
    Save the loss landscape for a single parameter to a file.
    
    Args:
        filename: Output filename for the plot
        param_idx: Index of the parameter to vary
        grid_points: Number of grid points
        param_range: Range to vary parameter around the base value
        theta_base: Base parameter values. If None, uses optimal_params or init_theta
    """
    
    if theta_base is None:
        if tqfm.optimal_params is not None:
            theta_base = tqfm.optimal_params.copy()
        elif tqfm.init_theta is not None:
            theta_base = tqfm.init_theta.copy()
        else:
            raise ValueError("No parameter values available. Please fit the model first or provide theta_base.")
    
    # Create parameter range
    param_center = theta_base[param_idx]
    param_vals = np.linspace(param_center - param_range, param_center + param_range, grid_points)
    loss_vals = np.zeros(grid_points)
    
    # Compute loss for each parameter value
    print(f"Computing 1D loss landscape ({grid_points} points)...")
    for i in range(grid_points):
        theta_temp = theta_base.copy()
        theta_temp[param_idx] = param_vals[i]
        loss_vals[i] = tqfm._loss(theta_temp, tqfm.X_train, tqfm.y_train, 
                                    tqfm.circuit, tqfm.num_classes, store_history=False)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(param_vals, loss_vals, linewidth=2, color='blue', label='Loss')
    ax.set_xlabel(f'θ[{param_idx}]', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'Loss Landscape for Parameter θ[{param_idx}]', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Mark initial point if available
    if tqfm.init_theta is not None:
        init_loss = tqfm._loss(theta_base, tqfm.X_train, tqfm.y_train, 
                                tqfm.circuit, tqfm.num_classes, store_history=False)
        ax.scatter([tqfm.init_theta[param_idx]], [init_loss],
                    color='yellow', s=150, marker='o', 
                    edgecolors='black', linewidths=2,
                    label='Initial', zorder=5)
    
    # Mark optimal point if available
    if tqfm.optimal_params is not None:
        ax.scatter([tqfm.optimal_params[param_idx]], [tqfm.optimal_value], 
                    color='red', s=200, marker='*', 
                    edgecolors='black', linewidths=2,
                    label=f'Optimal (Loss={tqfm.optimal_value:.4f})', zorder=5)
    
    ax.legend(fontsize=10, loc='best')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"1D loss landscape saved to {filename}")



def plot_landscape_2d(tqfm : TrainableQuantumFeatureMap, param_idx1: int = 0, param_idx2: int = 1, 
                        grid_points: int = 30, param_range: float = np.pi,
                        theta_base: np.ndarray = None):
    """
    Plot the loss landscape by varying two parameters.
    
    Args:
        param_idx1: Index of the first parameter to vary
        param_idx2: Index of the second parameter to vary
        grid_points: Number of grid points in each dimension
        param_range: Range to vary parameters around the base values
        theta_base: Base parameter values. If None, uses optimal_params or init_theta
    """
    
    if theta_base is None:
        if tqfm.optimal_params is not None:
            theta_base = tqfm.optimal_params.copy()
        elif tqfm.init_theta is not None:
            theta_base = tqfm.init_theta.copy()
        else:
            raise ValueError("No parameter values available. Please fit the model first or provide theta_base.")
    
    # Create parameter grids
    param1_center = theta_base[param_idx1]
    param2_center = theta_base[param_idx2]
    
    param1_vals = np.linspace(param1_center - param_range, param1_center + param_range, grid_points)
    param2_vals = np.linspace(param2_center - param_range, param2_center + param_range, grid_points)
    
    P1, P2 = np.meshgrid(param1_vals, param2_vals)
    Loss = np.zeros_like(P1)
    
    # Compute loss for each point in the grid
    print(f"Computing loss landscape ({grid_points}x{grid_points} grid)...")
    for i in range(grid_points):
        for j in range(grid_points):
            theta_temp = theta_base.copy()
            theta_temp[param_idx1] = P1[i, j]
            theta_temp[param_idx2] = P2[i, j]
            Loss[i, j] = tqfm._loss(theta_temp, tqfm.X_train, tqfm.y_train, 
                                    tqfm.circuit, tqfm.num_classes, store_history=False)
    
    # Create 3D surface plot
    fig = plt.figure(figsize=(12, 5))
    
    # 3D surface
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(P1, P2, Loss, cmap=cm.coolwarm, alpha=0.8)
    ax1.set_xlabel(f'θ[{param_idx1}]')
    ax1.set_ylabel(f'θ[{param_idx2}]')
    ax1.set_zlabel('Loss')
    ax1.set_title('Loss Landscape (3D)')
    
    # Mark optimal point if available
    if tqfm.optimal_params is not None:
        ax1.scatter([tqfm.optimal_params[param_idx1]], 
                    [tqfm.optimal_params[param_idx2]], 
                    [tqfm.optimal_value], 
                    color='red', s=100, marker='*', label='Optimal')
        ax1.legend()
    
    fig.colorbar(surf, ax=ax1, shrink=0.5)
    
    # 2D contour
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(P1, P2, Loss, levels=20, cmap=cm.coolwarm)
    ax2.contour(P1, P2, Loss, levels=20, colors='black', alpha=0.3, linewidths=0.5)
    ax2.set_xlabel(f'θ[{param_idx1}]')
    ax2.set_ylabel(f'θ[{param_idx2}]')
    ax2.set_title('Loss Landscape (Contour)')
    
    # Mark optimal point if available
    if tqfm.optimal_params is not None:
        ax2.scatter([tqfm.optimal_params[param_idx1]], 
                    [tqfm.optimal_params[param_idx2]], 
                    color='red', s=100, marker='*', label='Optimal')
        ax2.legend()
    
    fig.colorbar(contour, ax=ax2)
    plt.tight_layout()
    plt.show()


def save_landscape_2d(tqfm : TrainableQuantumFeatureMap, filename: str, param_idx1: int = 0, param_idx2: int = 1, 
                        grid_points: int = 30, param_range: float = np.pi,
                        theta_base: np.ndarray = None):
    """
    Save the loss landscape plot to a file.
    
    Args:
        filename: Output filename for the plot
        param_idx1: Index of the first parameter to vary
        param_idx2: Index of the second parameter to vary
        grid_points: Number of grid points in each dimension
        param_range: Range to vary parameters around the base values
        theta_base: Base parameter values. If None, uses optimal_params or init_theta
    """
    
    if theta_base is None:
        if tqfm.optimal_params is not None:
            theta_base = tqfm.optimal_params.copy()
        elif tqfm.init_theta is not None:
            theta_base = tqfm.init_theta.copy()
        else:
            raise ValueError("No parameter values available. Please fit the model first or provide theta_base.")
    
    # Create parameter grids
    param1_center = theta_base[param_idx1]
    param2_center = theta_base[param_idx2]
    
    param1_vals = np.linspace(param1_center - param_range, param1_center + param_range, grid_points)
    param2_vals = np.linspace(param2_center - param_range, param2_center + param_range, grid_points)
    
    P1, P2 = np.meshgrid(param1_vals, param2_vals)
    Loss = np.zeros_like(P1)
    
    # Compute loss for each point in the grid
    print(f"Computing loss landscape ({grid_points}x{grid_points} grid)...")
    for i in range(grid_points):
        for j in range(grid_points):
            theta_temp = theta_base.copy()
            theta_temp[param_idx1] = P1[i, j]
            theta_temp[param_idx2] = P2[i, j]
            Loss[i, j] = tqfm._loss(theta_temp, tqfm.X_train, tqfm.y_train, 
                                    tqfm.circuit, tqfm.num_classes, store_history=False)
    
    # Create 3D surface plot
    fig = plt.figure(figsize=(12, 5))
    
    # 3D surface
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(P1, P2, Loss, cmap=cm.coolwarm, alpha=0.8)
    ax1.set_xlabel(f'θ[{param_idx1}]')
    ax1.set_ylabel(f'θ[{param_idx2}]')
    ax1.set_zlabel('Loss')
    ax1.set_title('Loss Landscape (3D)')
    
    # Mark optimal point if available
    if tqfm.optimal_params is not None:
        ax1.scatter([tqfm.optimal_params[param_idx1]], 
                    [tqfm.optimal_params[param_idx2]], 
                    [tqfm.optimal_value], 
                    color='red', s=100, marker='*', label='Optimal')
        ax1.legend()
    
    fig.colorbar(surf, ax=ax1, shrink=0.5)
    
    # 2D contour
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(P1, P2, Loss, levels=20, cmap=cm.coolwarm)
    ax2.contour(P1, P2, Loss, levels=20, colors='black', alpha=0.3, linewidths=0.5)
    ax2.set_xlabel(f'θ[{param_idx1}]')
    ax2.set_ylabel(f'θ[{param_idx2}]')
    ax2.set_title('Loss Landscape (Contour)')
    
    # Mark optimal point if available
    if tqfm.optimal_params is not None:
        ax2.scatter([tqfm.optimal_params[param_idx1]], 
                    [tqfm.optimal_params[param_idx2]], 
                    color='red', s=100, marker='*', label='Optimal')
        ax2.legend()
    
    fig.colorbar(contour, ax=ax2)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Loss landscape saved to {filename}")



def plot_kernel_matrix(kernel_matrix, title="Quantum Kernel Matrix", filename=None, cmap='Greys'):
    """Plot the kernel matrix as a heatmap."""
    import matplotlib.pyplot as plt
    plt.imshow(kernel_matrix, cmap=cmap, interpolation='nearest')
    plt.colorbar(label='Kernel Value')
    plt.title(title)
    if filename:
        plt.savefig(filename)
    plt.show()


def plot_multi_kernel_matrices(matrices, titles, filename=None, cmap='Greys'):
    """Plot multiple kernel matrices side by side."""
    import matplotlib.pyplot as plt
    n = len(matrices)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    for i in range(n):
        ax = axes[i] if n > 1 else axes
        im = ax.imshow(matrices[i], cmap=cmap, interpolation='nearest')
        ax.set_title(titles[i])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if filename:
        plt.savefig(filename)
    plt.show()