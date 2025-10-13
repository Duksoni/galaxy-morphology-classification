import h5py
import keras
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.constants import N_CLASSES
from src.model.model_types import ModelType
from src.preprocessing.augmentation import Augmentation
from src.preprocessing.class_weights import get_class_weights
from src.preprocessing.preprocess import preprocess


def prepare_dataset(model_type: ModelType, augmentation: Augmentation, batch_size: int, seed: int, path: str = "data/Galaxy10_DECals.h5"):
    """
    Loads dataset from the specified path, splits it into train, val and test sets and preprocesses it.

    :param model_type: Model used for training
    :param augmentation: If augmentation should be applied to the dataset
    :param path: path to the dataset
    :param batch_size: Training batch size
    :param seed: Seed for random operations
    :return: Tuple containing train, val and test datasets and computed class weights, if augmentation is used
    """

    print("Loading dataset")

    with h5py.File(path, "r") as f:
        images = f["images"][:]
        labels = f["ans"][:]

    print("Splitting into train/val/test")

    train_val_idx, test_idx = train_test_split(np.arange(len(labels)), test_size=0.15, random_state=seed)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.176, random_state=seed)  # ~15% val

    print("Converting splits to tensorflow Datasets")

    labels_cat = keras.utils.to_categorical(labels, N_CLASSES).astype(np.float32)
    train_ds = tf.data.Dataset.from_tensor_slices((images[train_idx], labels_cat[train_idx]))
    val_ds = tf.data.Dataset.from_tensor_slices((images[val_idx], labels_cat[val_idx]))
    test_ds = tf.data.Dataset.from_tensor_slices((images[test_idx], labels_cat[test_idx]))

    class_weights = get_class_weights(N_CLASSES, labels[train_idx]) if augmentation != Augmentation.NONE else None
    if class_weights:
        print("Class weights:", class_weights)

    buffer_size = len(train_ds) + 1
    print(f"Shuffle buffer size: {buffer_size}")

    train_ds = (
        train_ds
        .map(lambda image, label: preprocess(model_type, image, label, True, augmentation),
             num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(buffer_size=buffer_size, seed=seed)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        val_ds
        .map(lambda image, label: preprocess(model_type, image, label),
             num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    test_ds = (
        test_ds
        .map(lambda image, label: preprocess(model_type, image, label), 
             num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    del images, labels, labels_cat, train_val_idx, test_idx, train_idx, val_idx
    return train_ds, val_ds, test_ds, class_weights
