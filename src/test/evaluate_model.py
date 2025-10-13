import keras
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from src.constants import CLASS_NAMES, CLASS_NAMES_BRIEF
from src.model.model_types import ModelType


def evaluate_model(test_dataset: tf.data.Dataset, model_path: str, model_type: ModelType, batch_size: int,
                   plot_matrix: bool = True):
    """
    Evaluates trained model on test dataset and prints classification report and confusion matrix.

    :param test_dataset: Preprocessed test dataset, returned by `prepare_dataset` function
    :param model_path: Path to the model file
    :param model_type: Model type for which the model was trained
    :param batch_size: Batch size for evaluation
    :param plot_matrix: If confusion matrix should be plotted
    :return: Tuple containing test accuracy, test loss, true labels and predicted labels
    """

    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path)
    test_loss, test_acc = model.evaluate(test_dataset, batch_size=batch_size)
    print(f"\n-----------------Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}-----------------\n")

    y_pred_probs = model.predict(test_dataset, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.concatenate([label.numpy() for _, label in test_dataset], axis=0)
    y_true = np.argmax(y_true, axis=1)

    print(f"\nClassification Report for {model_type.value}:\n",
          classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))
    if plot_matrix:
        matrix = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sb.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES_BRIEF, yticklabels=CLASS_NAMES)
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.show()

    return test_acc, test_loss, y_true, y_pred
