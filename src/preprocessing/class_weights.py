import numpy as np
import sklearn.utils


def get_class_weights(n_classes: int, labels):
    class_weights = sklearn.utils.compute_class_weight(
        class_weight="balanced",
        classes=np.arange(n_classes),  # explicitly 0..9
        y=labels  # integer labels
    )
    return dict(enumerate(class_weights))
