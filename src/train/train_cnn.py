import keras
import tensorflow as tf

import src.train.callbacks as callbacks

tb_callback = callbacks.tensorboard_callback("logs/cnn")


def train_cnn(
        model: keras.models.Sequential,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        class_weights: dict | None,
        n_epochs: int,
        file_path: str
) -> dict:
    return model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=n_epochs,
        class_weight=class_weights,
        callbacks=[
            callbacks.early_stopping_callback(),
            callbacks.checkpoint_callback(file_path),
            callbacks.lr_scheduler_callback(),
            tb_callback
        ]
    ).history
