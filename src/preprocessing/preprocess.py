import keras
import tensorflow as tf

from src import constants
from src.model.model_types import ModelType
from src.preprocessing.augmentation import Augmentation
from src.preprocessing.augmnet import augment_pipeline, stronger_augment_pipeline

pipleine = augment_pipeline()

stronger_pipeline = stronger_augment_pipeline()


def preprocess(model_type: ModelType, image, label, is_training: bool = False,
               augmentation: Augmentation = Augmentation.NONE):
    image = tf.image.resize(image, constants.IMAGE_SIZE)
    if model_type == ModelType.CNN:
        image = tf.cast(image, tf.float32) / 255.0
    elif model_type == ModelType.RESNET:
        image = keras.applications.resnet50.preprocess_input(image)
    else:
        raise ValueError(f"Invalid model type {model_type}")

    if is_training:
        if augmentation == Augmentation.NORMAL:
            image = pipleine(image, training=True)
        elif augmentation == Augmentation.STRONG:
            image = stronger_pipeline(image, training=True)

    return image, label
