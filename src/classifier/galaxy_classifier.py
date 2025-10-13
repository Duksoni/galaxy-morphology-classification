import tensorflow as tf

from src.model.cnn import build_cnn
from src.model.model_types import ModelType
from src.model.resnet import build_resnet50
from src.preprocessing.augmentation import Augmentation
from src.test.evaluate_model import evaluate_model
from src.train.prepare_dataset import prepare_dataset
from src.train.train_cnn import train_cnn
from src.train.train_history_graph import plot_training_history
from src.train.train_resnet import train_resnet50


class GalaxyClassifier:
    """
    Manages the classification of galaxies using various machine learning models.
    Encapsulates all the necessary logic for preparing dataset, trainin and evaluating the model.
    """

    def __init__(self, model_type: ModelType, file_path: str):
        """
        :param model_type: Model used for training
        :param file_path: Path where the model will be saved
        """
        self._model_type = model_type
        self._file_path = file_path
        self._model = None
        self._train_fn = None
        self._init_model()

    @property
    def model_type(self):
        return self._model_type

    @model_type.setter
    def model_type(self, model_type: ModelType):
        self._model_type = model_type
        self._init_model()

    @property
    def file_path(self):
        return self._file_path

    @file_path.setter
    def file_path(self, file_path: str):
        self._file_path = file_path

    def prepare_dataset(self, augmentation: Augmentation, batch_size: int, seed: int):
        if self._model is None:
            raise ValueError("Model not set")
        return prepare_dataset(self._model_type, augmentation, batch_size, seed)

    def train(self, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, weights: dict | None, max_epochs: int,
              plot_history: bool = True):
        if self._model is None or self._train_fn is None:
            raise ValueError("Model not set")
        history = self._train_fn(self._model, train_ds, val_ds, weights, max_epochs, self._file_path)
        if plot_history:
            plot_training_history(history, self._model_type)
        return history

    def evaluate(self, test_ds: tf.data.Dataset, batch_size: int, plot_matrix: bool = True):
        if self._model is None:
            raise ValueError("Model not set")
        return evaluate_model(test_ds, self._file_path, self._model_type, batch_size, plot_matrix)

    def _init_model(self):
        match self.model_type:
            case ModelType.CNN:
                self._model = build_cnn()
                self._train_fn = train_cnn
            case ModelType.RESNET:
                self._model = build_resnet50()
                self._train_fn = train_resnet50
