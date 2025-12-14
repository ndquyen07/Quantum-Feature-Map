import numpy as np

def class_similarity(kernel_matrix, y):
    # index của từng class
    unique_classes = np.unique(y)
    class_indices = {cls: np.where(y == cls)[0] for cls in unique_classes}

    # Between-class similarities
    between_similarities = []
    for i, cls1 in enumerate(unique_classes):
        for cls2 in unique_classes[i+1:]:
            idx1 = class_indices[cls1]
            idx2 = class_indices[cls2]
            between = kernel_matrix[np.ix_(idx1, idx2)].mean()
            between_similarities.append(between)

    average_between = np.mean(between_similarities) if between_similarities else 0

    # Separability ratio
    sep_ratio = 1 / average_between if average_between != 0 else np.inf

    return sep_ratio





