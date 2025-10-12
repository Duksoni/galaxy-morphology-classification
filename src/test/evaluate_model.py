import keras
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from src.constants import BATCH_SIZE
from src.model.model_types import ModelType


def evaluate_model(test_dataset: tf.data.Dataset, model_path: str, model_type: ModelType):
    """
    Evaluates trained model on test dataset and prints classification report and confusion matrix.

    :param test_dataset: Preprocessed test dataset, returned by `prepare_dataset` function
    :param model_path: Path to the model file
    :param model_type: Model type for which the model was trained
    """

    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path)
    test_loss, test_acc = model.evaluate(test_dataset, batch_size=BATCH_SIZE)
    print(f"\n-----------------Test Accuracy: {test_acc:.4f}, Loss: {test_loss:.4f}-----------------\n")

    y_pred_probs = model.predict(test_dataset, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.concatenate([label.numpy() for _, label in test_dataset], axis=0)
    y_true = np.argmax(y_true, axis=1)

    print(f"\nClassification Report for {model_type.value}:\n", classification_report(y_true, y_pred, digits=4))

    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sb.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()
